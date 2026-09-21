"""Measured, source-preserving launch-bounds and workspace Split-K candidates.

Split-K supports only the recognized dense, row-major, full-BK kernel protocol.
Unsupported kernels are recorded as skipped, never replaced by a generic GEMM.
"""
import copy
import hashlib
import math
import re
from pathlib import Path

from SGPO.utils.common_utils import save_json
from SGPO.verification.build_run_verifier import verify_build_and_run
from SGPO.verification.gemm_semantic_checker import extract_launch_config


def launch_bounds_source(source, blocks):
    c=extract_launch_config(source)
    threads=c['BM']*c['BN']//(c['WM']*c['WN'])*32
    if threads > 1024 or blocks not in (1,2):
        raise ValueError('unsupported launch configuration')
    updated,n=re.subn(r'__global__\s+void\s+gemm\s*\(',
                     f'__global__ __launch_bounds__({threads}, {blocks}) void gemm(',source)
    if n!=1:
        raise ValueError('kernel declaration is not a supported unannotated gemm')
    return updated


def split_k_source(source, ir, slices, reduce_threads=256, vector_width=4):
    p=ir.get('problem',{})
    c=extract_launch_config(source)
    if any(p.get('layout_'+x)!='row_major' or p.get('trans_'+x) is not False for x in ('A','B')):
        raise ValueError('requires non-transposed row-major inputs')
    if any(not isinstance(p.get(d),int) or not c.get(t) or p[d]%c[t] for d,t in [('M','BM'),('N','BN'),('K','BK')]):
        raise ValueError('requires whole M/N/K tiles; no partial K slice protocol')
    tiles=p['K']//c['BK']
    if not 1<slices<=tiles or reduce_threads not in (128,256) or vector_width not in (1,4):
        raise ValueError('unsupported reduction parameters')
    if vector_width==4 and p['M']*p['N']%4:
        raise ValueError('vector reduce requires a whole vector output extent')
    if p['M']*p['N']*4*slices>256*1024*1024:
        raise ValueError('workspace exceeds 256 MiB budget')
    if 'valid_k' in source or 'cp.async' in source or 'sgpo_slices' in source:
        raise ValueError('kernel needs a dedicated slice-aware tail/async transformation')
    signature=r'(__global__\s+void\s+gemm\s*\([\s\S]*?float\s*\*C\s*)\)\s*\{'
    preamble=f''') {{
    const int sgpo_first = (K / BK) * blockIdx.z / sgpo_slices;
    const int sgpo_count = (K / BK) * (blockIdx.z + 1) / sgpo_slices - sgpo_first;
    A += sgpo_first * BK;
    B += sgpo_first * BK * N;
    C += static_cast<size_t>(blockIdx.z) * M * N;
    alpha = 1.0f;
    beta = 0.0f;
'''
    source,n=re.subn(signature,lambda m:m[1]+', int sgpo_slices'+preamble,source)
    if n!=1:
        raise ValueError('unsupported kernel signature')
    source,n=re.subn(r'const int k_tiles = K / BK;', 'const int k_tiles = sgpo_count;',source)
    if n!=1:
        raise ValueError('requires a single explicit full-BK tile loop')
    source,n=re.subn(r'void cuda_gemm\(', 'void sgpo_partial_dispatch(',source)
    if n!=1:
        raise ValueError('unsupported host entry')
    source,n=re.subn(r'<<<blocksPerGrid, threadsPerBlock>>>\(M, N, K, alpha, A, B, beta, C\);',
                     f'<<<dim3(blocksPerGrid.x, blocksPerGrid.y, {slices}), threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C, {slices});',source)
    if n!=1:
        raise ValueError('unsupported launch site')
    # beta=0 partial sums must not read uninitialized workspace.
    source,n=re.subn(r'\+ beta \* C\[c_index\]', '+ (beta == 0.0f ? 0.0f : beta * C[c_index])',source)
    if n<1 or 'if (beta == 1.0f)' in source:
        raise ValueError('unsupported specialized epilogue; use the general alpha/beta seed')
    reduce_body = '''float4 sum = make_float4(0,0,0,0);
    for (int s=0;s<S;++s) {
        float4 v = reinterpret_cast<const float4*>(workspace + static_cast<size_t>(s)*size)[idx];
        sum.x+=v.x; sum.y+=v.y; sum.z+=v.z; sum.w+=v.w;
    }
    float4 old = beta == 0 ? make_float4(0,0,0,0) : reinterpret_cast<float4*>(C)[idx];
    reinterpret_cast<float4*>(C)[idx] = make_float4(alpha*sum.x+beta*old.x, alpha*sum.y+beta*old.y, alpha*sum.z+beta*old.z, alpha*sum.w+beta*old.w);''' if vector_width==4 else '''float sum=0;
    for (int s=0;s<S;++s) sum+=workspace[static_cast<size_t>(s)*size+idx];
    C[idx]=alpha*sum+(beta==0 ? 0 : beta*C[idx]);'''
    return '#include <cstdio>\n#include <cstdlib>\n'+source+f'''
__global__ void sgpo_reduce(const float* workspace, float* C, int size, int S, float alpha, float beta) {{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=size/{vector_width}) return;
    {reduce_body}
}}
void cuda_gemm(int M,int N,int K,float alpha,float* A,float* B,float beta,float* C) {{
    float* workspace=nullptr;
    cudaError_t status=cudaMallocAsync(reinterpret_cast<void**>(&workspace), static_cast<size_t>({slices})*M*N*sizeof(float), 0);
    if (status!=cudaSuccess) {{ fprintf(stderr,"CUDA error: %s\\n",cudaGetErrorString(status)); exit(1); }}
    sgpo_partial_dispatch(M,N,K,1.0f,A,B,0.0f,workspace);
    sgpo_reduce<<<CEIL_DIV(M*N/{vector_width},{reduce_threads}),{reduce_threads}>>>(workspace,C,M*N,{slices},alpha,beta);
    status=cudaFreeAsync(workspace,0);
    if (status!=cudaSuccess) {{ fprintf(stderr,"CUDA error: %s\\n",cudaGetErrorString(status)); exit(1); }}
}}
'''


def run_bounded_unlock(candidates, output, build_platform='windows', benchmark_runs=5, benchmark_warmup_runs=2,
                       load_layout_trials=True):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    results, records=[],[]
    for seed_index,seed in enumerate(candidates[:3],1):
        snapshot=seed['source_snapshot']
        original=snapshot['cuda_kernel.cuh']
        specs=[('control',None)] + [(f'launch_{b}',('launch',b)) for b in (1,2)]
        if load_layout_trials:
            from SGPO.verification.load_layout_trials import TRIALS
            specs += [('load_layout_' + name, ('load_layout', name)) for name in TRIALS]
        specs += [(f'split_{s}_r256_v4',('split',s,256,4)) for s in (2,3,4,6)]
        # Explore reduction geometry on the best measured split, not all combinations.
        best_split=None
        for name,spec in specs:
            folder=output/f'seed{seed_index}'/name
            record={'seed':seed_index,'variant':name,'code_dir':str(folder)}
            layout = {}
            try:
                if spec and spec[0] == 'load_layout':
                    from SGPO.verification.load_layout_trials import transform
                    source, layout = transform(original, spec[1])
                else:
                    source=original if spec is None else (launch_bounds_source(original,spec[1]) if spec[0]=='launch'
                        else split_k_source(original,seed['verified_ir'],*spec[1:]))
            except (ValueError,KeyError) as exc:
                record.update(status='skipped',reason=str(exc))
                records.append(record)
                continue
            folder.mkdir(parents=True)
            for file,content in snapshot.items():
                if file in ('main.cpp','kernel.h','cuda_kernel.cuh'):
                    (folder/file).write_text(source if file=='cuda_kernel.cuh' else content,encoding='utf-8')
            ir=copy.deepcopy(seed['verified_ir'])
            ir.pop('verification',None)
            ir.pop('performance',None)
            from SGPO.verification.optimization_preservation import active_source
            async_in_source=bool(re.search(r'cp\.async|cuda::memcpy_async|__pipeline_memcpy_async',active_source(source)))
            ir.setdefault('pipeline',{})['async_copy']=async_in_source
            ir.setdefault('compiler',{}).setdefault('resource_feedback',{})['enabled']=False
            sid=None
            superseded = []
            if spec and spec[0] == 'load_layout':
                from SGPO.verification.load_layout_trials import update_trial_ir
                superseded = update_trial_ir(ir, spec[1], layout)
                record['explicit_layout_replacement'] = copy.deepcopy(ir['load_layout_trial'])
            elif spec:
                sid='Compiler.LaunchBounds.MinBlocksPerSM' if spec[0]=='launch' else 'Reduction.SplitK.Workspace'
                ir.setdefault('strategy',{}).setdefault('applied_strategy_ids',[]).append(sid)
                if spec[0]=='launch':
                    ir['compiler']['launch_bounds_min_blocks']=spec[1]
                    tile=extract_launch_config(source)
                    ir['compiler']['launch_bounds_max_threads']=tile['BM']*tile['BN']//(tile['WM']*tile['WN'])*32
                else:
                    ir.setdefault('scheduling',{})['work_decomposition']='split_k_workspace'
                    ir['reduction']={'split_k_slices':spec[1],'reduce_threads':spec[2],'vector_width':spec[3],
                                     'workspace_required':True,'finalization':'separate_kernel',
                                     'workspace_bytes':4*ir['problem']['M']*ir['problem']['N']*spec[1]}
            verified=verify_build_and_run(ir,folder,folder/'build',build_platform=build_platform,
                                          benchmark_runs=benchmark_runs,benchmark_warmup_runs=benchmark_warmup_runs)
            passed=all(verified.get('verification',{}).get(k,{}).get('status')=='pass'
                       for k in ('compile','correctness','runtime_safety'))
            speed=verified.get('performance',{}).get('gflops')
            passed=passed and isinstance(speed,(int,float)) and math.isfinite(speed) and speed>0
            summary={k+'_status':verified['verification'].get(k,{}).get('status') for k in ('compile','correctness','runtime_safety')}
            summary.update(cuda_error=verified['verification'].get('runtime_safety',{}).get('cuda_error'),
                           gflops=verified.get('performance',{}).get('gflops'),latency_ms=verified.get('performance',{}).get('latency_ms'))
            summary.update(oracle_acceptance_status='pass' if passed else 'fail',
                           acceptance_basis='compile_correctness_runtime_oracle',
                           code_verification_status='not_run',
                           code_verification_basis='bounded_deterministic_transform_plus_runtime_checks')
            verified['verification']['summary']=summary
            # Realization belongs to this source, never to the inherited seed.
            from SGPO.verification.strategy_realization_oracle import verify_strategy_realization
            from SGPO.verification.strategy_application import strategy_application, update_strategy_contract_state
            verified['verification']['accepted'] = passed
            verified['strategy_realization'] = verify_strategy_realization(
                folder, verified.get('strategy', {}).get('applied_strategy_ids', []), verified)
            verified['strategy_application'] = strategy_application(verified, [sid] if sid else [])
            update_strategy_contract_state(verified)
            record.update(status='pass' if passed else 'fail',accepted=passed,verification=summary,
                          source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          timing_scope='entire cuda_gemm call: allocation, partial GEMM, reduction, release')
            save_json(folder/'verified_ir.json',verified)
            save_json(folder/'candidate.json',record)
            records.append(record)
            candidate=copy.deepcopy(seed)
            candidate.update(accepted=passed,verified_ir=verified,candidate_code_dir=str(folder),
                             source_snapshot={**snapshot,'cuda_kernel.cuh':source},
                             strategy_id=seed['strategy_id']+'.'+name,source_phase='deterministic_resource_unlock')
            if spec and spec[0] == 'load_layout':
                candidate['path'] = [s for s in seed.get('path', []) if s not in superseded] + ['MeasuredLoadLayout:' + spec[1]]
                candidate['history'] = copy.deepcopy(verified.get('strategy', {}))
                candidate['source_phase'] = 'deterministic_load_layout_unlock'
            if sid:
                candidate['path']=list(seed.get('path',[]))+[sid]
            results.append(candidate)
            if passed and spec and spec[0]=='split' and spec[2]==256:
                if best_split is None or summary['gflops']>best_split[0]:
                    best_split=(summary['gflops'],spec[1])
            if name=='split_6_r256_v4' and best_split:
                specs.extend([(f'split_{best_split[1]}_r128_v4',('split',best_split[1],128,4)),
                              (f'split_{best_split[1]}_r256_v1',('split',best_split[1],256,1))])
            save_json(output/'report.json',{'records':records})
            print(f"resource unlock seed={seed_index} {name}: {summary}",flush=True)
    save_json(output/'report.json',{'records':records})
    return results
