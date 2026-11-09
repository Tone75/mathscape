"""Independent saved-only RQ-M08 allocation audit; no primary engine or RNG."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import stats,special

ROOT=Path(__file__).resolve().parents[1]
DISTRIBUTIONS=('gamma','lognormal')
METHODS=('t','percentile','bca')
ALLOCATIONS={'P30':(4,30),'P60':(2,60),'P120':(1,120)}
O,R,STREAMS,N,B=10,20,120,20,4999
MAIN_NAMESPACE=0x4D3038
ACCEPTED_INPUT='39bcdc367320b6da513df72d2fa43bfdb7f060f4'
NUMERIC_RTOL,NUMERIC_ATOL=2e-10,2e-12
IDENTITY_RTOL,IDENTITY_ATOL=2e-12,2e-12
CONFIDENCE=.995
ALLOCATION_METRICS=('T','Vbar','lower_sd','upper_sd','S','Ase','Apop','Vse','C','Mbelow','Mabove','miss_T','mean_width','warning_rate')
GROUP_FIELDS=('lower_mean','upper_mean','lower_var','upper_var','V','A')
DECOMPOSITION_FIELDS=('Vbar30','T30','T60','T120','Jgap','Pgap','delta','residual','normalized_residual','abs_normalized_residual')+tuple(f'ss{part}_{side}_{field}' for part in (30,60) for side in ('lower','upper') for field in ('residual','scale','normalized_residual','abs_normalized_residual'))

def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sample_hash(sample):
    return hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes(order="C")).hexdigest()


def _sd(values):
    values = np.asarray(values, dtype=float)
    return 0.0 if np.all(values == values[0]) else float(np.std(values, ddof=1))


def _original_se(samples):
    """Reconstruct early sample rejection before a rounded variance is used.

    Exactly constant fractional values can produce tiny positive np.std values.
    Such samples were rejected before inference, so their saved SE is undefined.
    """
    samples = np.asarray(samples, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        se = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
        rejected = (~np.isfinite(samples).all(axis=1) | np.all(samples == samples[:, :1], axis=1)
                    | ~np.isfinite(np.mean(samples, axis=1)))
    se[rejected] = np.nan
    return se


def mean_interval(values, confidence=0.995):
    """Independent-original mean +/- Student critical value; no clipping."""
    values = np.asarray(values, dtype=float)
    total = values.size
    output = dict(N=total, finite_count=int(np.isfinite(values).sum()), mean=None, sd=None,
                  mcse=None, low=None, high=None, df=total - 1, defined=False)
    if total == 0 or not np.all(np.isfinite(values)):
        return output
    output["mean"] = float(np.mean(values))
    if total < 2:
        return output
    mean, sd = float(np.mean(values)), _sd(values)
    output.update(mean=mean, sd=sd, mcse=sd / np.sqrt(total))
    if sd <= 0:
        return output
    half = stats.t.ppf((1 + confidence) / 2, total - 1) * output["mcse"]
    return output | dict(low=float(mean - half), high=float(mean + half), defined=True)


def welch_interval(first, second, confidence=0.995, *, min_group=2):
    """Difference first-minus-second at the independent original-sample level."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    output = dict(N1=first.size, N0=second.size, finite_count1=int(np.isfinite(first).sum()),
                  finite_count0=int(np.isfinite(second).sum()), mean1=None, mean0=None, variance1=None, variance0=None,
                  difference=None, mcse=None, df=None,
                  low=None, high=None, defined=False)
    if min(first.size, second.size) < max(2, min_group) or not (np.isfinite(first).all() and np.isfinite(second).all()):
        return output
    difference = float(np.mean(first) - np.mean(second))
    first_var, second_var = _sd(first) ** 2, _sd(second) ** 2
    output.update(mean1=float(np.mean(first)), mean0=float(np.mean(second)),
                  variance1=float(first_var), variance0=float(second_var))
    parts = np.array([first_var / first.size, second_var / second.size])
    variance = float(parts.sum())
    denominator = parts[0] ** 2 / (first.size - 1) + parts[1] ** 2 / (second.size - 1)
    output["difference"] = difference
    if variance <= 0 or denominator <= 0:
        return output
    mcse, df = np.sqrt(variance), variance**2 / denominator
    half = stats.t.ppf((1 + confidence) / 2, df) * mcse
    return output | dict(mcse=float(mcse), df=float(df), low=float(difference - half),
                         high=float(difference + half), defined=True)


def _meta(panel, name):
    value = np.asarray(panel[name])
    if value.ndim != 0:
        raise ValueError(f"metadata {name} must be scalar")
    return value.item()


def compare_rows(expected, observed, key_fields, *, label="table"):
    """Compare every independently calculable field, including NA and counts."""
    errors = []

    def key(row):
        return tuple(str(row[field]) for field in key_fields)

    expected_index = {key(row): row for row in expected}
    observed_index = {key(row): row for row in observed}
    if len(expected_index) != len(expected):
        errors.append(f"{label}: duplicate independently reconstructed keys")
    if len(observed_index) != len(observed):
        errors.append(f"{label}: duplicate saved keys")
    if expected_index.keys() != observed_index.keys():
        errors.append(f"{label}: missing/extra keys ({len(expected_index)} expected; {len(observed_index)} saved)")
    for row_key in expected_index.keys() & observed_index.keys():
        actual = observed_index[row_key]
        for field, reference in expected_index[row_key].items():
            if field not in actual:
                errors.append(f"{label}/{row_key}: missing column {field}")
                continue
            value = actual[field]
            if isinstance(reference, (bool, np.bool_)):
                matches = str(value).lower() in (("true", "1") if reference else ("false", "0"))
            elif reference is None or isinstance(reference, (int, float, np.integer, np.floating)):
                target, number = finite_number(reference), finite_number(value)
                if target is None:
                    matches = number is None
                elif number is None:
                    matches = False
                elif isinstance(reference, (int, np.integer)):
                    matches = number == target
                else:
                    matches = np.isclose(number, target, atol=2e-12, rtol=2e-10)
            else:
                matches = str(value) == str(reference)
            if not matches:
                errors.append(f"{label}/{row_key}: {field} differs")
    return errors


def compare_json(expected, observed, *, location="hypotheses"):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [f"{location}: expected object"]
        for key, value in expected.items():
            if key not in observed:
                errors.append(f"{location}: missing {key}")
            else:
                errors.extend(compare_json(value, observed[key], location=f"{location}/{key}"))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return [f"{location}: list length/type differs"]
        for index, (first, second) in enumerate(zip(expected, observed)):
            errors.extend(compare_json(first, second, location=f"{location}/{index}"))
    else:
        mismatch = compare_rows([dict(id=0, value=expected)], [dict(id=0, value=observed)], ["id"], label=location)
        if mismatch:
            errors.append(f"{location}: value differs")
    return errors


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reason_array(panel):
    codes, labels = panel["reason_codes"], panel["reason_labels"].astype(str)
    if codes.dtype.kind not in "ui" or np.any(codes < 0) or np.any(codes >= labels.size):
        raise ValueError("reason code is outside the saved dictionary")
    return labels[codes]


def reference_intervals(sample, means):
    """Independent fixed-array reference; no statistical model imports or RNG."""
    sample, means = np.asarray(sample, dtype=float), np.asarray(means, dtype=float)
    empty = dict(lower=np.nan, upper=np.nan, z0=np.nan, a=np.nan, q_low=np.nan, q_high=np.nan,
                 tau_low=np.nan, tau_high=np.nan, valid=False, tail_warning=False)
    results = {method: empty.copy() for method in METHODS}
    if sample.shape != (20,) or not np.isfinite(sample).all() or np.all(sample == sample[0]):
        return results
    mean, se = float(np.mean(sample)), float(_original_se(sample[None, :])[0])
    if not np.isfinite(se) or se <= 0:
        return results
    jackknife = (sample.sum() - sample) / 19
    centered = jackknife.mean() - jackknife
    a = float(np.sum(centered**3) / (6 * np.sum(centered**2) ** 1.5))
    critical = stats.t.ppf(0.975, 19)
    results["t"].update(lower=mean - critical * se, upper=mean + critical * se, valid=True, a=a)
    results["percentile"]["a"] = results["bca"]["a"] = a
    if means.ndim != 1 or means.size == 0 or not np.isfinite(means).all():
        return results

    def record(qlo, qhi, **diagnostics):
        if not np.isfinite([qlo, qhi]).all() or not 0 <= qlo < qhi <= 1:
            return empty | diagnostics
        lower, upper = np.quantile(means, [qlo, qhi], method="linear")
        low_tau, high_tau = means.size * min(qlo, 1 - qlo), means.size * min(qhi, 1 - qhi)
        return empty | dict(lower=float(lower), upper=float(upper), valid=True, q_low=float(qlo), q_high=float(qhi),
                            tau_low=low_tau, tau_high=high_tau, tail_warning=min(low_tau, high_tau) < 10) | diagnostics

    results["percentile"] = record(0.025, 0.975, a=a)
    probability = (np.count_nonzero(means < mean) + 0.5 * np.count_nonzero(means == mean)) / means.size
    z0 = float(special.ndtri(probability))
    if not np.isfinite([a, z0]).all():
        results["bca"].update(a=a, z0=z0)
        return results
    z = stats.norm.ppf([0.025, 0.975])
    denominator = 1 - a * (z0 + z)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        results["bca"].update(a=a, z0=z0)
        return results
    q = special.ndtr(z0 + (z0 + z) / denominator)
    results["bca"] = record(q[0], q[1], a=a, z0=z0)
    return results




def _variance(values):
    values=np.asarray(values,dtype=float)
    return 0. if np.all(values==values[0]) else float(np.sum((values-values.mean())**2)/(values.size-1))


def _identity(residual,scale):return abs(residual)<=IDENTITY_ATOL+IDENTITY_RTOL*scale


def budget_metrics(lower,upper,valid,warning,se):
    """One budget repeat, three allocations sharing all saved streams exactly."""
    lower,upper,valid,warning=(np.asarray(value) for value in (lower,upper,valid,warning));streams=lower.shape[1]
    if lower.shape!=(3,streams) or streams%4 or streams//4<2:raise ValueError('three methods and four equal endpoint groups required')
    observed=valid[0]&np.isfinite(lower[0])&np.isfinite(upper[0])
    if observed.any() and (not np.all(lower[0][observed]==lower[0][observed][0]) or not np.all(upper[0][observed]==upper[0][observed][0])):raise ValueError('observed t endpoints vary across streams')
    methods=[]
    for mi,method in enumerate(METHODS):
        with np.errstate(over='ignore',invalid='ignore'):width=upper[mi]-lower[mi]
        mask=valid[mi]&np.isfinite(lower[mi])&np.isfinite(upper[mi])&np.isfinite(width)&(lower[mi]<=upper[mi])
        total=1 if method=='t' else streams;count=int(mask.all()) if method=='t' else int(mask.sum())
        initial=[]
        for allocation,j in (('P30',4),('P60',2),('P120',1)):
            k=streams//j
            initial.append(dict(allocation=allocation,J=j,K=k,**dict.fromkeys(ALLOCATION_METRICS,None),groups=[dict(group=group,group_start=group*k,group_stop=(group+1)*k,**dict.fromkeys(GROUP_FIELDS,np.nan)) for group in range(j)]))
        base=dict(method=method,complete=False,total_inner=total,valid_inner=count,invalid_inner=total-count,allocations=initial,diagnostics=dict.fromkeys(DECOMPOSITION_FIELDS,None),algebra_ok=False)
        if not np.isfinite(se) or se<=0 or not mask.all():
            base['reason']='sample_se_invalid' if not np.isfinite(se) or se<=0 else 'endpoint_invalid_or_nonfinite';methods.append(base);continue
        lo,hi=lower[mi],upper[mi];allocations=[]
        for allocation,j in (('P30',4),('P60',2),('P120',1)):
            k=streams//j;group_rows=[]
            for group in range(j):
                a,b=group*k,(group+1)*k;ls,us=_variance(lo[a:b]),_variance(hi[a:b])
                if method=='t':ls=us=v=0.
                else:
                    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):v=float(np.divide(ls+us,2*np.float64(se)*np.float64(se)))
                group_rows.append(dict(group=group,group_start=a,group_stop=b,lower_mean=float(np.mean(lo[a:b])),upper_mean=float(np.mean(hi[a:b])),lower_var=ls,upper_var=us,V=v,A=float(np.sqrt(v))))
            T=float(np.mean([row['A'] for row in group_rows]));vbar=float(np.mean([row['V'] for row in group_rows]))
            lsd=float(np.mean([np.sqrt(row['lower_var']) for row in group_rows]));usd=float(np.mean([np.sqrt(row['upper_var']) for row in group_rows]))
            S=float(np.mean([np.hypot(np.sqrt(row['lower_var']),np.sqrt(row['upper_var']))/np.sqrt(2) for row in group_rows]))
            descriptive=dict(lower_sd=lsd,upper_sd=usd,S=S,Ase=T,Apop=S*np.sqrt(20),Vse=vbar,C=float(np.mean((lo<=0)&(hi>=0))),Mbelow=float(np.mean(hi<0)),Mabove=float(np.mean(lo>0)),miss_T=float(np.mean(hi<0)-np.mean(lo>0)),mean_width=float(np.mean(hi-lo)),warning_rate=float(np.mean(warning[mi])))
            allocations.append(dict(allocation=allocation,J=j,K=k,T=T,Vbar=vbar,groups=group_rows,**descriptive))
        vbar=allocations[0]['Vbar'];sqrtvbar=float(np.sqrt(vbar));t30,t120=allocations[0]['T'],allocations[2]['T']
        jgap=sqrtvbar-t30;pgap=t120-sqrtvbar;delta=t120-t30;residual=delta-jgap-pgap;dscale=max(abs(delta),abs(jgap+pgap))
        diagnostics=dict(Vbar30=vbar,T30=t30,T60=allocations[1]['T'],T120=t120,Jgap=jgap,Pgap=pgap,delta=delta,residual=residual,normalized_residual=residual/max(1.,dscale),abs_normalized_residual=abs(residual)/max(1.,dscale))
        algebra=_identity(residual,dscale) and jgap>=-(IDENTITY_ATOL+IDENTITY_RTOL*max(abs(sqrtvbar),abs(t30)))
        for allocation in allocations[:2]:
            j,k=allocation['J'],allocation['K']
            for side,values in (('lower',lo),('upper',hi)):
                lhs=(streams-1)*_variance(values);groups=allocation['groups']
                rhs=(k-1)*sum(row[side+'_var'] for row in groups)+k*sum((row[side+'_mean']-values.mean())**2 for row in groups)
                raw,scale=float(lhs-rhs),float(max(abs(lhs),abs(rhs)))
                if method=='t':raw=scale=0.
                key=f'ss{30 if j==4 else 60}_{side}'
                diagnostics.update({key+'_residual':raw,key+'_scale':scale,key+'_normalized_residual':raw/max(1.,scale),key+'_abs_normalized_residual':abs(raw)/max(1.,scale)})
                algebra=algebra and _identity(raw,scale)
        if method=='t':
            for key in DECOMPOSITION_FIELDS:diagnostics[key]=0.
        finite=all(np.isfinite([row['T'],row['Vbar']]).all() and all(np.isfinite(list(item.values())).all() for item in row['groups']) for row in allocations)
        base.update(complete=bool(algebra and finite),reason='derived_statistic_nonfinite' if not finite else 'algebraic_identity_failed' if not algebra else '',allocations=allocations if algebra and finite else initial,diagnostics=diagnostics if finite else dict.fromkeys(DECOMPOSITION_FIELDS,None),algebra_ok=bool(algebra and finite))
        methods.append(base)
    return methods


def original_metrics(panel):
    output=[];se=_original_se(panel['samples'])
    for oi,outer in enumerate(panel['outer_id']):
        tlo,thi=panel['lower'][oi,:,0],panel['upper'][oi,:,0];mask=panel['valid'][oi,:,0]&np.isfinite(tlo)&np.isfinite(thi)
        if mask.any() and (not np.all(tlo[mask]==tlo[mask][0]) or not np.all(thi[mask]==thi[mask][0])):raise ValueError('observed t endpoints vary across budget repeats')
        repeats=[budget_metrics(panel['lower'][oi,r],panel['upper'][oi,r],panel['valid'][oi,r],panel['tail_warning'][oi,r],se[oi]) for r in range(panel['lower'].shape[1])]
        for mi,method in enumerate(METHODS):
            measured=[rep[mi] for rep in repeats];complete=len(measured)==R and all(rep['complete'] for rep in measured)
            total=1 if method=='t' else R*STREAMS;count=int(all(rep['valid_inner']==1 for rep in measured) and len(measured)==R) if method=='t' else sum(rep['valid_inner'] for rep in measured)
            row=dict(distribution=_meta(panel,'distribution'),master_seed=_meta(panel,'master_seed'),outer_id=int(outer),method=method,se=float(se[oi]),sample_sha256=sample_hash(panel['samples'][oi]),complete=bool(complete),planned_repeats=R,complete_repeats=sum(rep['complete'] for rep in measured),total_inner=total,valid_inner=count,invalid_inner=total-count,repeats=measured,
                     M30=None,M60=None,M120=None,Q30=None,Q60=None,Q120=None,d=None,L=None,U=None,ratio_Q120_Q30=None)
            if complete:
                values={name:np.array([rep['allocations'][pi]['T'] for rep in measured]) for pi,name in enumerate(('30','60','120'))}
                for name,array in values.items():row['M'+name]=float(array.mean());row['Q'+name]=_variance(array)
                row.update(d=float(np.mean(values['120']-values['30'])),L=row['Q120']-.8*row['Q30'],U=row['Q120']-1.25*row['Q30'],ratio_Q120_Q30=row['Q120']/row['Q30'] if row['Q30']>0 else None)
            output.append(row)
    return output


def mean_status(interval,complete=True):
    if not complete or not interval['defined']:return 'inconclusive'
    lo,hi=interval['low'],interval['high']
    if lo>-.0025 and hi<.0025:return 'supported'
    if lo>.0025 or hi<-.0025:return 'not_supported'
    return 'inconclusive'


def variance_status(lower,upper,mean_q30,complete=True):
    if not complete or not lower['defined'] or not upper['defined'] or finite_number(mean_q30) is None or mean_q30<=0:return 'inconclusive'
    if lower['low']>0 and upper['high']<0:return 'supported'
    if lower['high']<=0 or upper['low']>=0:return 'not_supported'
    return 'inconclusive'


def joint_status(statuses):
    statuses=list(statuses)
    return 'not_supported' if 'not_supported' in statuses else 'supported' if statuses==['supported','supported'] else 'inconclusive'


def inspect_panel(panel,strict_design=True):
    errors=[]
    check=lambda condition,message:errors.append(message) if not condition else None
    samples=panel["samples"];r,n=samples.shape
    j=panel["lower"].shape[1];k=panel["lower"].shape[3]
    shape=(r,j,3,k)
    check(samples.dtype==np.dtype('float64') and panel['sample_sha256'].dtype==np.dtype('S64'),'original sample/hash dtype differs')
    check(panel['outer_id'].dtype==np.dtype('uint16') and panel['budget_repeat_id'].dtype==np.dtype('uint8') and panel['stream_id'].dtype==np.dtype('uint8'),'ID dtype differs')
    check(panel['reason_codes'].dtype==np.dtype('uint8') and panel['reason_labels'].dtype.kind=='U' and len(panel['reason_labels'])<=256,'reason dictionary dtype/size differs')
    for field in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):
        check(panel[field].dtype==np.dtype('float64'),f'{field} dtype differs')
    if 'observed_streams' in panel:
        check(panel['observed_streams'].shape==(r,j,k) and panel['observed_streams'].dtype.kind=='b','observed stream shape/dtype differs')
        check(panel['observed_streams'].all(),'uncomputed required streams remain')
    elif strict_design:errors.append('observed stream mask missing')
    check(n==20 and _meta(panel,"n")==20 and _meta(panel,"alpha")==.05 and _meta(panel,"B")==4999,"fixed n/alpha/B differ")
    check(np.array_equal(panel["outer_id"],np.arange(r)),"outer IDs differ from fixed prefix")
    check(np.array_equal(panel["budget_repeat_id"],np.arange(j)),"budget repeat IDs differ")
    check(np.array_equal(panel["stream_id"],np.arange(k)),"stream IDs differ")
    check(list(panel["methods"].astype(str))==list(METHODS),"method axis differs")
    if strict_design:
        check((r,j,k)==(10,20,120) and _meta(panel,"O")==10 and _meta(panel,"R")==20 and _meta(panel,"streams_per_budget")==120,"main O/R/streams differ")
        check(_meta(panel,"namespace")==MAIN_NAMESPACE and _meta(panel,"distribution") in DISTRIBUTIONS
              and _meta(panel,"master_seed") in (0,1,2),"main namespace/population/seed differ")
        check(_meta(panel,"input_accepted_head")==ACCEPTED_INPUT,"accepted input head differs")
    for name in ("lower","upper","a","z0","q_low","q_high","tau_low","tau_high","valid","tail_warning","reason_codes"):
        check(panel[name].shape==shape,f"{name} shape differs from common original/replicate/method/inner grid")
    if errors:return errors
    valid,lo,hi=panel["valid"],panel["lower"],panel["upper"];reason=_reason_array(panel)
    check(valid.dtype.kind=="b" and panel["tail_warning"].dtype.kind=="b","mask dtype differs")
    check(np.isfinite(lo[valid]).all() and np.isfinite(hi[valid]).all() and np.all(lo[valid]<=hi[valid]),"valid endpoints nonfinite/reversed")
    check(np.isnan(lo[~valid]).all() and np.isnan(hi[~valid]).all(),"invalid endpoints substituted")
    check(np.all(reason[valid]=="") and np.all(reason[~valid]!=""),"reason/valid dictionary differs")
    hashes=panel["sample_sha256"].astype(str)
    check(hashes.shape==(r,) and all(hashes[i]==sample_hash(sample) for i,sample in enumerate(samples)),"original hash mismatch")
    se=_original_se(samples);original_valid=np.isfinite(se)&(se>0);critical=stats.t.ppf(.975,19)
    for oi,sample in enumerate(samples):
        observed=valid[oi,:,0]&np.isfinite(lo[oi,:,0])&np.isfinite(hi[oi,:,0])
        if not original_valid[oi]:
            check(not observed.any(),f"outer{oi}: invalid original has a valid t endpoint")
            continue
        mean=np.mean(sample)
        observed_lo,observed_hi=lo[oi,:,0][observed],hi[oi,:,0][observed]
        check(np.allclose(observed_lo,mean-critical*se[oi],atol=1e-12,rtol=1e-12)
              and np.allclose(observed_hi,mean+critical*se[oi],atol=1e-12,rtol=1e-12),f"outer{oi}: t reference differs")
        check(not observed_lo.size or (np.all(observed_lo==observed_lo[0]) and np.all(observed_hi==observed_hi[0])),f"outer{oi}: observed t varies by replicate/inner")
        jackknife=(sample.sum()-sample)/19;deviations=jackknife.mean()-jackknife
        a=np.sum(deviations**3)/(6*np.sum(deviations**2)**1.5)
        stored=panel["a"][oi];finite=np.isfinite(stored)
        check(np.allclose(stored[finite],a,atol=1e-12,rtol=1e-10),f"outer{oi}: acceleration differs from independent jackknife")
        check(np.isfinite(panel["a"][oi,:,2][valid[oi,:,2]]).all(),f"outer{oi}: valid BCa acceleration undefined")
    for mi,method in enumerate(METHODS):
        mask=valid[:,:,mi]
        if method=="t":
            check(not panel["tail_warning"][:,:,mi].any(),"t has bootstrap tail warning");continue
        qlo,qhi=panel["q_low"][:,:,mi],panel["q_high"][:,:,mi]
        check(np.isfinite(qlo[mask]).all() and np.isfinite(qhi[mask]).all()
              and np.all((qlo[mask]>=0)&(qlo[mask]<qhi[mask])&(qhi[mask]<=1)),f"{method}: invalid quantiles")
        if method=="percentile":
            check(np.allclose(qlo[mask],.025) and np.allclose(qhi[mask],.975),"percentile nominal q differ")
        else:
            z0,a=panel["z0"][:,:,mi][mask],panel["a"][:,:,mi][mask]
            check(np.isfinite(z0).all(),"BCa valid z0 nonfinite")
            ranks=2*4999*special.ndtr(z0)
            check(np.allclose(ranks,np.rint(ranks),atol=2e-8,rtol=0),"BCa z0 off half-tie rank grid")
            for probability,observed in ((.025,qlo[mask]),(.975,qhi[mask])):
                z=stats.norm.ppf(probability);den=1-a*(z0+z)
                expected=special.ndtr(z0+(z0+z)/den)
                check(np.all(den>0) and np.allclose(observed,expected,atol=2e-12,rtol=1e-10),"BCa adjusted quantile formula differs")
        l_tau,u_tau=4999*np.minimum(qlo,1-qlo),4999*np.minimum(qhi,1-qhi)
        check(np.allclose(panel["tau_low"][:,:,mi][mask],l_tau[mask],atol=1e-10,rtol=1e-11)
              and np.allclose(panel["tau_high"][:,:,mi][mask],u_tau[mask],atol=1e-10,rtol=1e-11),f"{method}: tau differs")
        check(np.array_equal(panel["tail_warning"][:,:,mi][mask],((l_tau<10)|(u_tau<10))[mask]),f"{method}: warning differs")
    return errors


def audit_stream(path,panel,dist,seed,inner):
    errors=[];check=lambda condition,message:errors.append(message) if not condition else None
    with np.load(path,allow_pickle=False) as source:stream={key:source[key] for key in source.files}
    indices,means=stream["indices"],stream["means"]
    check(indices.shape==(4999,20) and indices.dtype==np.dtype("uint8"),"indices shape/dtype differ")
    check(means.shape==(4999,) and means.dtype==np.dtype("float64"),"means shape/dtype differ")
    if errors:return errors
    check(np.all(indices<20),"index outside source sample")
    if np.any(indices>=20):return errors
    for name,value in dict(distribution=dist,master_seed=seed,outer_id=0,budget_repeat_id=0,stream_id=inner,n=20,B=4999,R=20,streams_per_budget=120,
                           alpha=.05,namespace=MAIN_NAMESPACE,native_indices_dtype="int64",stored_indices_dtype="uint8").items():
        check(_meta(stream,name)==value,f"stream {name} metadata differ")
    digest=_meta(stream,"sample_sha256");digest=digest.decode("ascii") if isinstance(digest,bytes) else str(digest)
    sample=panel["samples"][0]
    check(digest==sample_hash(sample),"stream original hash differs")
    expected_means=np.mean(sample[indices],axis=1)
    check(np.array_equal(expected_means,means),"saved means differ from indices/original")
    reference=reference_intervals(sample,expected_means)
    for mi,method in enumerate(METHODS):
        actual={field:panel[field][0,0,mi,inner].item() for field in reference[method]}
        errors.extend(compare_rows([dict(id=0,**reference[method])],[dict(id=0,**actual)],["id"],label=method))
    return errors




def _stream_counts(panel,oi,mi,repeat=None,start=0,stop=120,whole_required_t=True):
    lower=panel['lower'][oi,:,mi,start:stop] if repeat is None else panel['lower'][oi,repeat,mi,start:stop]
    upper=panel['upper'][oi,:,mi,start:stop] if repeat is None else panel['upper'][oi,repeat,mi,start:stop]
    valid=panel['valid'][oi,:,mi,start:stop] if repeat is None else panel['valid'][oi,repeat,mi,start:stop]
    with np.errstate(over='ignore',invalid='ignore'):width=upper-lower
    mask=valid&np.isfinite(lower)&np.isfinite(upper)&np.isfinite(width)&(lower<=upper)
    if mi==0 and whole_required_t and (start!=0 or stop!=panel['lower'].shape[3]):return _stream_counts(panel,oi,mi,repeat,0,panel['lower'].shape[3])
    total=1 if mi==0 else mask.size;count=int(mask.all()) if mi==0 else int(mask.sum())
    return dict(total_streams=total,valid_streams=count,invalid_streams=total-count)


def derived_tables(panel):
    """Independently flatten all seven groups, three T values and original units."""
    tables={key:[] for key in ('original_allocations','original_comparisons','budgets','groups','decompositions')}
    original=original_metrics(panel);streams=panel['lower'].shape[3];repeats=panel['lower'].shape[1]
    for row in original:
        oi=row['outer_id'];mi=METHODS.index(row['method']);method=row['method']
        meta={key:row[key] for key in ('distribution','master_seed','outer_id','method','se','sample_sha256')}
        meta.update(n=20,originals_per_cell=int(_meta(panel,'O')),R=repeats,streams_per_budget=streams,B=4999,alpha=.05)
        counts=_stream_counts(panel,oi,mi,stop=streams)
        base=dict(complete=row['complete'],complete_repeats=row['complete_repeats'],planned_repeats=R,**counts)
        comp={**meta,**base,**{key:row[key] for key in ('d','L','U','Q30','Q120','M30','M120','ratio_Q120_Q30')}}
        names=('Jgap','Pgap','delta');residuals=('residual','normalized_residual')+tuple(f'ss{p}_{side}_{field}' for p in (30,60) for side in ('lower','upper') for field in ('residual','normalized_residual'))
        for field in names:comp['mean_'+field]=float(np.mean([rep['diagnostics'][field] for rep in row['repeats']])) if row['complete'] else None
        for field in residuals:comp['max_abs_'+field]=max(abs(rep['diagnostics'][field]) for rep in row['repeats']) if row['complete'] else None
        tables['original_comparisons'].append(comp)
        for pi,(allocation,(j,k)) in enumerate(ALLOCATIONS.items()):
            data={metric:float(np.mean([rep['allocations'][pi][metric] for rep in row['repeats']])) if row['complete'] else None for metric in ALLOCATION_METRICS}
            kval=streams//j;M=row['M'+allocation[1:]];Q=row['Q'+allocation[1:]]
            tables['original_allocations'].append({**meta,'allocation':allocation,'J':j,'K':kval,**base,'M':M,'Q':Q,'Q_over_R':Q/R if Q is not None else None,**data})
        for repeat,rep in enumerate(row['repeats']):
            counts=_stream_counts(panel,oi,mi,repeat=repeat,stop=streams)
            diag={**meta,'budget_repeat_id':repeat,**counts,'complete':rep['complete'],'algebra_ok':rep['algebra_ok'],'reason':rep['reason'],**rep['diagnostics']}
            diag.update({gap+'_sign':int(np.sign(diag[gap])) if finite_number(diag[gap]) is not None else 'NA' for gap in names});tables['decompositions'].append(diag)
            for pi,allocation in enumerate(rep['allocations']):
                allocmeta={**meta,'budget_repeat_id':repeat,'allocation':allocation['allocation'],'J':allocation['J'],'K':allocation['K']}
                tables['budgets'].append({**allocmeta,'complete':rep['complete'],**counts,**{metric:allocation[metric] for metric in ALLOCATION_METRICS}})
                for group in allocation['groups']:
                    index=group['group'] if pi==0 else group['group']+4 if pi==1 else 6
                    groupcounts=_stream_counts(panel,oi,mi,repeat=repeat,start=group['group_start'],stop=group['group_stop'])
                    tables['groups'].append({**allocmeta,'group':group['group'],'group_index':index,'group_start':group['group_start'],'group_stop':group['group_stop'],
                                             'complete':rep['complete'],**groupcounts,**{metric:group[metric] for metric in GROUP_FIELDS}})
    return tables


def _expected(rows,dist,seed,method,metric,allocation=None):
    group=[row for row in rows if row['distribution']==dist and row['master_seed']==seed and row['method']==method and (allocation is None or row['allocation']==allocation)]
    index={row['outer_id']:row for row in group}
    if len(index)!=len(group) or not set(index).issubset(range(O)):raise ValueError('duplicate/out-of-range original ID')
    return np.array([index[i][metric] if i in index and index[i]['complete'] else np.nan for i in range(O)],dtype=float)


def _values(rows,dist,seed,method,metric,allocation=None):
    seeds=(0,1,2) if seed=='pooled' else (seed,)
    return np.concatenate([_expected(rows,dist,master,method,metric,allocation) for master in seeds])


def _interval_record(values):
    result=mean_interval(values);defined=result.pop('defined');result['valid']=defined
    result['reason']='' if defined else 'values_nonfinite' if result['finite_count']!=result['N'] else 'too_few_originals' if result['N']<2 else 'variance_zero_or_nonfinite'
    return result


def metric_statistics(values):
    values=np.asarray(values,dtype=float);interval=mean_interval(values)
    result=dict(mean=interval['mean'],sd=interval['sd'],mcse=interval['mcse'],decision_low=interval['low'],decision_high=interval['high'])
    quantiles=np.quantile(values,[.05,.25,.5,.75,.95],method='linear') if len(values) and np.isfinite(values).all() else [None]*5
    result.update({f'p{int(q*100):02d}':finite_number(value) for q,value in zip((.05,.25,.5,.75,.95),quantiles)})
    return result


def summary_rows(original_allocations,pooled=False):
    output=[]
    for dist in DISTRIBUTIONS:
        for seed in (['pooled'] if pooled else (0,1,2)):
            seeds=(0,1,2) if seed=='pooled' else (seed,);planned=O*len(seeds)
            for method in METHODS:
                for allocation,(j,k) in ALLOCATIONS.items():
                    group=[row for row in original_allocations if row['distribution']==dist and row['master_seed'] in seeds and row['method']==method and row['allocation']==allocation];complete=sum(row['complete'] for row in group)
                    row=dict(distribution=dist,master_seed=seed,n=20,R=20,allocation=allocation,J=j,K=k,streams_per_budget=120,B=4999,alpha=.05,method=method,
                             original_total=planned,observed_original_count=len(group),missing_original_count=planned-len(group),complete_original_count=complete,incomplete_original_count=planned-complete,
                             conditional_original_count=complete,interval_count=sum(item['total_streams'] for item in group),valid_count=sum(item['valid_streams'] for item in group),invalid_count=sum(item['invalid_streams'] for item in group))
                    for metric in ('M','Q','Q_over_R',*ALLOCATION_METRICS):
                        values=_values(original_allocations,dist,seed,method,metric,allocation)
                        for prefix,array in (('',values),('conditional_',values[np.isfinite(values)])):
                            for suffix,value in metric_statistics(array).items():row[prefix+metric+('' if suffix=='mean' else '_'+suffix)]=value
                    output.append(row)
    return output


def hypothesis_reference(comparisons):
    quantities={};decisions={};seed_details=[]
    for dist in DISTRIBUTIONS:
        intervals={}
        for H,field in (('H1','d'),('H2_L','L'),('H2_U','U')):
            values=_values(comparisons,dist,'pooled','bca',field);interval=mean_interval(values);intervals[H]=interval
            quantities[H+'_'+dist]={**_interval_record(values),'field':field,'population':dist,'conditional_interval':_interval_record(values[np.isfinite(values)])}
        h1=mean_status(intervals['H1']);reason='' if h1!='inconclusive' else 'incomplete_or_undefined_primary_comparison' if not intervals['H1']['defined'] else 'margin_touched_or_crossed'
        quantities['H1_'+dist].update(status=h1,reason=reason)
        q=_values(comparisons,dist,'pooled','bca','Q30');qmean=float(q.mean()) if np.isfinite(q).all() else None
        h2=variance_status(intervals['H2_L'],intervals['H2_U'],qmean)
        defined=intervals['H2_L']['defined'] and intervals['H2_U']['defined'] and qmean is not None and qmean>0
        h2reason='' if h2!='inconclusive' else 'incomplete_or_undefined_primary_comparison' if not defined else 'multiplicative_band_touched_or_crossed'
        decisions[dist]=dict(H1=dict(status=h1,reason=reason),H2=dict(status=h2,reason=h2reason,mean_Q30=qmean,lower_interval='H2_L_'+dist,upper_interval='H2_U_'+dist))
    for seed in (0,1,2):seed_details.append(dict(master_seed=seed,**{f'{H}_{dist}':_interval_record(_expected(comparisons,dist,seed,'bca',field)) for dist in DISTRIBUTIONS for H,field in (('H1','d'),('H2_L','L'),('H2_U','U'))}))
    return dict(question_id='RQ-M08',confirmatory_intervals=6,decision_confidence=.995,independent_originals_per_population=30,target_method='bca',B=4999,R=20,
                allocations=[dict(allocation=p,J=j,K=k) for p,(j,k) in ALLOCATIONS.items()],primary_comparison='P120-P30',P60='descriptive only',seed_direction_gate=False,H1_margin=.0025,H1_units='Ase units,not percentagepoints',H2_variance_ratio_band=[.8,1.25],
                quantities=quantities,population_decisions=decisions,seed_details=seed_details,hypotheses={H:dict(status=joint_status([decisions[d][H]['status'] for d in DISTRIBUTIONS]),**{d:decisions[d][H]['status'] for d in DISTRIBUTIONS}) for H in ('H1','H2')})


def decomposition_rows(comparisons,decompositions):
    output=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2,'pooled'):
            seeds=(0,1,2) if seed=='pooled' else (seed,)
            for method in METHODS:
                group=[row for row in decompositions if row['distribution']==dist and row['master_seed'] in seeds and row['method']==method]
                row=dict(distribution=dist,master_seed=seed,method=method,original_total=O*len(seeds),diagnostic_budget_repeats=len(group),interpretation='r-level signs/quantiles descriptive; original-level intervals N10/30; no M07Rgap')
                for gap in ('Jgap','Pgap','delta'):
                    finite=np.array([float(item[gap]) for item in group if finite_number(item[gap]) is not None])
                    row.update({gap+'_finite_count':len(finite),gap+'_negative_count':int((finite<0).sum()),gap+'_zero_count':int((finite==0).sum()),gap+'_positive_count':int((finite>0).sum()),gap+'_raw_mean':float(finite.mean()) if len(finite) else None})
                    quantiles=np.quantile(finite,[.05,.25,.5,.75,.95],method='linear') if len(finite) else [None]*5
                    for q,value in zip((.05,.25,.5,.75,.95),quantiles):row[f'{gap}_raw_p{int(q*100):02d}']=finite_number(value)
                    values=_values(comparisons,dist,seed,method,'mean_'+gap)
                    for prefix,array in (('',values),('conditional_',values[np.isfinite(values)])):
                        for key,value in _interval_record(array).items():row[f'{prefix}{gap}_original_{key}']=value
                for field in sorted(key for key in DECOMPOSITION_FIELDS if 'residual' in key):
                    finite=[abs(float(item[field])) for item in group if finite_number(item[field]) is not None];row[field+'_max_abs']=max(finite) if finite else None
                row.update(algebra_failed_repeats=sum(item['reason']=='algebraic_identity_failed' for item in group),algebra_unchecked_repeats=sum(not item['algebra_ok'] and item['reason']!='algebraic_identity_failed' for item in group),algebra_passed_repeats=sum(bool(item['algebra_ok']) for item in group))
                ratio=_values(comparisons,dist,seed,method,'ratio_Q120_Q30');q30=_values(comparisons,dist,seed,method,'Q30');q120=_values(comparisons,dist,seed,method,'Q120');finite=np.isfinite(ratio);paired=np.isfinite(q30)&np.isfinite(q120)
                row.update(ratio_planned_originals=len(ratio),ratio_finite_originals=int(finite.sum()),ratio_missing_or_undefined_originals=int((~finite).sum()),
                           mean_individual_Q120_Q30=float(ratio.mean()) if finite.all() else None,conditional_mean_individual_Q120_Q30=float(ratio[finite].mean()) if finite.any() else None,
                           ratio_mean_Q120_over_mean_Q30=float(q120.mean()/q30.mean()) if paired.all() and q30.mean()>0 else None,
                           conditional_ratio_mean_Q120_over_mean_Q30=float(q120[paired].mean()/q30[paired].mean()) if paired.any() and q30[paired].mean()>0 else None,
                           conditional_ratio_of_means_originals=int(paired.sum()),ratio_reference='descriptive; mean individual ratios differs from ratio meanQ; H2 uses meanL/U; zero denominatorNA')
                output.append(row)
    return output


def check_source_hashes(manifest,root=ROOT):
    errors=[];mapping=manifest.get('source_files_sha256',{})
    if 'scripts/run_allocation_guarded.py' not in mapping:errors.append('guard source hash missing')
    for name,digest in mapping.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:errors.append('source hash path outside project');continue
        try:data=(root/path).read_bytes()
        except OSError:errors.append(f'{name}: preregistered source missing');continue
        if hashlib.sha256(data).hexdigest()!=digest:errors.append(f'{name}: preregistered source hash differs')
    return errors


def check_inputs(manifest,panels,root=ROOT):
    errors=[];inputs=manifest['inputs'];check=lambda condition,message:errors.append(message) if not condition else None
    expected=dict(schema_version=1,question_id='RQ-M08',accepted_head=ACCEPTED_INPUT,source_question_id='RQ-M07',n=20,originals_per_cell=10,selected_originals=60,
                  m06_accepted_head='e2feae23397d014f6aa3363773da9913da85fcf7',m05_accepted_head='dc2501c3099f5c25227d9eee00d227c2f0b02f81',sample_hash_contract='SHA256 little-endian float64 C-order bytes')
    errors.extend(compare_json(expected,inputs,location='inputs'))
    sources={}
    for mapping_key,head,question in (('source_files_sha256',ACCEPTED_INPUT,'rq_m07'),('m06_files_sha256',expected['m06_accepted_head'],'rq_m06'),('m05_files_sha256',expected['m05_accepted_head'],'rq_m05')):
        paths={f'evidence/{question}/manifest.json',*(f'evidence/{question}/panels/{dist}_seed{seed}.npz' for dist in DISTRIBUTIONS for seed in (0,1,2))}
        mapping=inputs[mapping_key];check(set(mapping)==paths,f'{mapping_key}: expected exact manifest+six panel paths')
        for name,digest in mapping.items():
            if name not in paths:continue
            data=(root/name).read_bytes();check(hashlib.sha256(data).hexdigest()==digest,f'{name}: accepted file hash differs')
            committed=subprocess.check_output(['git','show',f'{head}:{name}'],cwd=root);check(data==committed,f'{name}: differs from accepted Git blob')
        sources[question]=json.loads((root/f'evidence/{question}/manifest.json').read_text())
    oldinputs=sources['rq_m07']['inputs']
    check(oldinputs['source_files_sha256']==inputs['m06_files_sha256'] and oldinputs['upstream_files_sha256']==inputs['m05_files_sha256'],'M07 manifest input chain differs')
    check(sources['rq_m07']['question_id']=='RQ-M07' and sources['rq_m06']['question_id']=='RQ-M06' and sources['rq_m05']['question_id']=='RQ-M05','source question chain differs')
    selections=inputs['selections'];check(len(selections)==6 and {(item['distribution'],item['master_seed']) for item in selections}==set(panels),'fixed input six selections differ')
    for selection in selections:
        key=selection['distribution'],selection['master_seed'];dist,seed=key
        if key not in panels:errors.append('required selected panel missing');continue
        originals=[]
        for question,path_key,size,namespace in (('rq_m07','source_path',10,0x4D3037),('rq_m06','m06_source_path',200,0x4D3036),('rq_m05','m05_source_path',10000,0x4D3035)):
            expected_path=f'evidence/{question}/panels/{dist}_seed{seed}.npz';check(selection[path_key]==expected_path,'selected source path differs from fixed cell')
            with np.load(root/expected_path,allow_pickle=False) as saved:old={name:saved[name] for name in saved.files}
            check(old['samples'].shape==(size,20) and old['samples'].dtype==np.dtype('<f8') and old['sample_sha256'].dtype==np.dtype('S64'),'source sample/hash shape or dtype differs')
            check(np.array_equal(old['outer_id'],np.arange(size)) and old['outer_id'].dtype==np.dtype('uint16'),'source fixed original IDs differ')
            check(_meta(old,'n')==20 and _meta(old,'R')==size and _meta(old,'alpha')==.05 and _meta(old,'namespace')==namespace and _meta(old,'distribution')==dist and _meta(old,'master_seed')==seed,'source fixed metadata differs')
            check(all(sample_hash(sample)==digest for sample,digest in zip(old['samples'],old['sample_sha256'].astype(str))),'source original row hash differs')
            if question=='rq_m07':check(_meta(old,'J')==20 and _meta(old,'K')==120 and _meta(old,'B')==4999 and _meta(old,'input_accepted_head')==expected['m06_accepted_head'],'M07 original lineage metadata differs')
            elif question=='rq_m06':check(_meta(old,'K')==30 and np.array_equal(old['B_values'],[999,4999]) and _meta(old,'input_accepted_head')==expected['m05_accepted_head'],'M06 original lineage metadata differs')
            else:check(_meta(old,'K')==1 and _meta(old,'B')==4999,'M05 K1/B metadata differs')
            originals.append(old['samples'][:10])
        sample=panels[key]['samples'];raw=np.asarray(sample,dtype='<f8').tobytes()
        check(all(raw==np.asarray(saved,dtype='<f8').tobytes() for saved in originals),'M08/M07/M06/M05 selected original bytes differ')
        hashes=[sample_hash(row) for row in sample]
        check(selection['outer_ids']==list(range(10)) and selection['sample_sha256']==hashes and selection['selected_samples_sha256']==sample_hash(sample),'fixed selected IDs/row/matrix hashes differ')
    return errors


def _read_table(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def audit_evidence(directory,*,root=ROOT):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text());errors=[];panels={};hashes={}
    tables={key:[] for key in ('original_allocations','original_comparisons','budgets','groups','decompositions')}
    design=dict(distributions=list(DISTRIBUTIONS),n=20,master_seeds=[0,1,2],originals_per_cell=10,R=20,streams_per_budget=120,
                allocations=[dict(allocation=p,J=j,K=k) for p,(j,k) in ALLOCATIONS.items()],B=4999,alpha=.05,outer_ids=list(range(10)),methods=list(METHODS),input_accepted_head=ACCEPTED_INPUT,source_question='RQ-M07')
    errors.extend(compare_json(dict(schema_version=1,question_id='RQ-M08',design=design),manifest,location='manifest'))
    tolerance=dict(numeric_rtol=NUMERIC_RTOL,numeric_atol=NUMERIC_ATOL,identity_rtol=IDENTITY_RTOL,identity_atol=IDENTITY_ATOL)
    errors.extend(compare_json(tolerance,manifest.get('numeric_tolerances',{}),location='numeric_tolerances'))
    errors.extend(compare_json(dict(wall_seconds=300,internal_stop_seconds=294,peak_rss_bytes=2*1024**3,output_bytes=50*1024**2,max_resample_elements=1000000,decision_confidence=.995,confirmatory_intervals=6,mean_movement_margin=.0025,variance_ratio_low=.8,variance_ratio_high=1.25),manifest.get('policy',{}),location='resource_policy'))
    errors.extend(compare_json(dict(main_namespace=MAIN_NAMESPACE,smoke_namespace=0x534D3038,distribution_ids=dict(gamma=1,lognormal=2),spawn_key=['namespace','distribution_id','n','outer_id','purpose_id','budget_repeat_id','stream_id'],purpose_ids=dict(bootstrap=1),budget_repeat_ids=list(range(20)),stream_ids=list(range(120)),outer_ids=list(range(10)),native_indices_dtype='int64',stored_indices_dtype='uint8'),manifest.get('randomness',{}),location='randomness'))
    errors.extend(check_source_hashes(manifest,root=root))
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            path=directory/'panels'/f'{dist}_seed{seed}.npz'
            if not path.exists():errors.append(f'missing panel {dist}/{seed}');continue
            hashes[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as saved:panel={name:saved[name] for name in saved.files}
            panels[(dist,seed)]=panel;errors.extend(f'{dist}/{seed}: {error}' for error in inspect_panel(panel))
            local=derived_tables(panel)
            for name,rows in local.items():tables[name].extend(rows)
    if len(list((directory/'panels').glob('*.npz')))!=6:errors.append('saved panel count differs from six')
    errors.extend(check_inputs(manifest,panels,root=root))
    alloc,comp=tables['original_allocations'],tables['original_comparisons']
    comparisons=[('original_allocation_metrics.csv',alloc,['distribution','master_seed','outer_id','method','allocation']),
                 ('original_comparison_metrics.csv',comp,['distribution','master_seed','outer_id','method']),
                 ('budget_metrics.csv.gz',tables['budgets'],['distribution','master_seed','outer_id','budget_repeat_id','method','allocation']),
                 ('group_metrics.csv.gz',tables['groups'],['distribution','master_seed','outer_id','budget_repeat_id','method','group_index']),
                 ('decomposition_metrics.csv.gz',tables['decompositions'],['distribution','master_seed','outer_id','budget_repeat_id','method']),
                 ('summary.csv',summary_rows(alloc),['distribution','master_seed','method','allocation']),
                 ('pooled_summary.csv',summary_rows(alloc,True),['distribution','master_seed','method','allocation']),
                 ('decomposition_summary.csv',decomposition_rows(comp,tables['decompositions']),['distribution','master_seed','method'])]
    counts={}
    for name,expected,keys in comparisons:
        observed=_read_table(directory/name);counts[name]=dict(reconstructed=len(expected),saved=len(observed));errors.extend(compare_rows(expected,observed,keys,label=name))
    errors.extend(compare_json(hypothesis_reference(comp),json.loads((directory/'hypotheses.json').read_text())))
    reconstructed=0
    for (dist,seed),panel in panels.items():
        for stream in (0,29,30,59,60,119):
            path=directory/'streams'/f'{dist}_seed{seed}_outer0_repeat0_stream{stream}.npz'
            if not path.exists():errors.append(f'missing designated stream {dist}/{seed}/{stream}');continue
            hashes[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
            errors.extend(f'{path.name}: {error}' for error in audit_stream(path,panel,dist,seed,stream));reconstructed+=1
    if len(list((directory/'streams').glob('*.npz')))!=36:errors.append('saved stream count differs from36')
    errors.extend(compare_json(dict(status='completed',completed_cells=6,incomplete_cells=[]),json.loads((directory/'resources.json').read_text()),location='resources'))
    return dict(question_id='RQ-M08',status='failed' if errors else 'passed',mismatch_count=len(errors),mismatches=errors,original_samples=sum(len(panel['samples']) for panel in panels.values()),table_counts=counts,
                fully_reconstructed_streams=reconstructed,other_generation_sequences_not_reconstructed=144000-36,numeric_tolerances=tolerance,saved_file_sha256=hashes,
                limitations=['Only36 predesignated stored-index streams reconstruct sampling/means/inference; remaining143964 generation sequences are not reconstructed.',
                             'All saved endpoint reductions, algebra, original-unit summaries and six main99.5% intervals independently reconstructed; not external scientific review.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path,nargs='?',default=ROOT/'evidence/rq_m08');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit_evidence(args.directory)
    if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],mismatch_count=result['mismatch_count'],table_counts=result['table_counts']),allow_nan=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
