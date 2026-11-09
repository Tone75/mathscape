"""Independent RQ-M07 saved-endpoint and designated-index audit.

No primary models, aggregation/decision functions or RNG are imported. Numerical
comparison and algebra tolerances are fixed here before the scientific run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import stats, special

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTIONS=("gamma","lognormal")
METHODS=("t","percentile","bca")
R,J,K,GROUP=10,20,120,30
B=4999
MAIN_NAMESPACE=0x4D3037
ACCEPTED_INPUT="e2feae23397d014f6aa3363773da9913da85fcf7"
NUMERIC_RTOL,NUMERIC_ATOL=2e-10,2e-12
IDENTITY_RTOL,IDENTITY_ATOL=2e-12,2e-12

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


def mean_interval(values, confidence=0.99):
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


def welch_interval(first, second, confidence=0.99, *, min_group=2):
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


BASE_METRICS=("lower_sd","upper_sd","S","Ase","Apop","Vse","C","Mbelow","Mabove","T","mean_width","warning_rate")


def _mean(values):
    values=np.asarray(values,dtype=float)
    return float(values[0]) if np.all(values==values[0]) else float(np.mean(values))


def _variance(values):
    values=np.asarray(values,dtype=float)
    return 0.0 if np.all(values==values[0]) else float(np.sum((values-np.mean(values))**2)/(values.size-1))


def _near_identity(residual,scale):
    return abs(residual)<=IDENTITY_ATOL+IDENTITY_RTOL*scale


def replicate_metrics(lower,upper,valid,warning,se):
    """A single J replicate, preserving first group versus four-group average."""
    lower,upper,valid,warning=(np.asarray(value) for value in (lower,upper,valid,warning))
    observed=valid[0]&np.isfinite(lower[0])&np.isfinite(upper[0])
    if observed.any() and (not np.all(lower[0][observed]==lower[0][observed][0]) or not np.all(upper[0][observed]==upper[0][observed][0])):
        raise ValueError("observed t endpoints vary across inner streams")
    group_size=lower.shape[1]//4
    result=[]
    for mi,method in enumerate(METHODS):
        mask=valid[mi]&np.isfinite(lower[mi])&np.isfinite(upper[mi])&(lower[mi]<=upper[mi])&np.isfinite(upper[mi]-lower[mi])
        count=1 if method=="t" else mask.size
        actual=np.array([mask.all()]) if method=="t" else mask
        complete=bool(np.all(mask) and np.isfinite(se) and se>0 and group_size>=2)
        row=dict(method=method,complete=complete,total_inner=count,valid_inner=int(actual.sum()),invalid_inner=int(count-actual.sum()),
                 group_lower_mean=np.full(4,np.nan),group_upper_mean=np.full(4,np.nan),group_lower_var=np.full(4,np.nan),
                 group_upper_var=np.full(4,np.nan),group_V=np.full(4,np.nan),group_A=np.full(4,np.nan),
                 by_K=[dict.fromkeys(BASE_METRICS,None) for _ in range(2)])
        scalar_names=("decomposition_normalized_residual","decomposition_abs_normalized_residual","lower_ss_normalized_residual","upper_ss_normalized_residual","lower_ss_abs_normalized_residual","upper_ss_abs_normalized_residual","V30","A30","V120","A120","Vbar","Abar","Jgap","Pgap","Rgap","delta",
                      "decomposition_residual","lower_ss_residual","upper_ss_residual","lower_ss_scale","upper_ss_scale")
        row.update(dict.fromkeys(scalar_names,None));row["algebra_ok"]=False
        if not complete:
            result.append(row);continue
        lo,hi=lower[mi],upper[mi]
        group_lo=lo.reshape(4,group_size);group_hi=hi.reshape(4,group_size)
        lmean=np.array([_mean(group) for group in group_lo]);umean=np.array([_mean(group) for group in group_hi])
        lvar=np.array([_variance(group) for group in group_lo]);uvar=np.array([_variance(group) for group in group_hi])
        vgroup=(lvar+uvar)/(2*se**2)
        afirst=float(np.sqrt(vgroup[0]));vfull=(_variance(lo)+_variance(hi))/(2*se**2)
        afull=float(np.sqrt(vfull));vbar=float(vgroup.mean());abar=float(np.sqrt(vgroup).mean())
        jgap=float(np.sqrt(vbar)-abar);pgap=float(afull-np.sqrt(vbar));rgap=float(abar-afirst);delta=afull-afirst
        residual=delta-(jgap+pgap+rgap)
        ss=[]
        for values,means,variances in ((lo,lmean,lvar),(hi,umean,uvar)):
            lhs=(values.size-1)*_variance(values)
            rhs=(group_size-1)*variances.sum()+group_size*np.sum((means-_mean(values))**2)
            ss.append((float(lhs-rhs),float(max(abs(lhs),abs(rhs)))))
        if method=="t":
            # Deterministic control; no numerical np.var noise from repeated floats.
            vgroup[:]=0;lvar[:]=0;uvar[:]=0
            afirst=vfull=afull=vbar=abar=jgap=pgap=rgap=delta=residual=0.
            ss=[(0.,0.),(0.,0.)]
        dscale=float(max(abs(delta),abs(jgap+pgap+rgap)))
        algebra=bool(_near_identity(residual,dscale) and all(_near_identity(res,scale) for res,scale in ss)
                     and jgap>=-(IDENTITY_ATOL+IDENTITY_RTOL*max(abs(np.sqrt(vbar)),abs(abar))))
        row.update(group_lower_mean=lmean,group_upper_mean=umean,group_lower_var=lvar,group_upper_var=uvar,
                   group_V=vgroup,group_A=np.sqrt(vgroup),V30=float(vgroup[0]),A30=afirst,V120=float(vfull),A120=afull,
                   Vbar=vbar,Abar=abar,Jgap=jgap,Pgap=pgap,Rgap=rgap,delta=delta,decomposition_residual=residual,
                   lower_ss_residual=ss[0][0],upper_ss_residual=ss[1][0],lower_ss_scale=ss[0][1],upper_ss_scale=ss[1][1],
                   algebra_ok=algebra,complete=algebra,decomposition_scale=dscale)
        for name,res,scale in (("decomposition",residual,dscale),("lower_ss",ss[0][0],ss[0][1]),("upper_ss",ss[1][0],ss[1][1])):
            row[f"{name}_normalized_residual"]=res/max(1.,scale)
            row[f"{name}_abs_normalized_residual"]=abs(res)/max(1.,scale)
        if algebra:
            for ki,size in enumerate((group_size,lo.size)):
                sublo,subhi=lo[:size],hi[:size]
                lsd,usd=(0.,0.) if method=="t" else (_sd(sublo),_sd(subhi))
                s=float(np.hypot(lsd,usd)/np.sqrt(2));below=float(np.mean(subhi<0));above=float(np.mean(sublo>0))
                row["by_K"][ki]=dict(lower_sd=lsd,upper_sd=usd,S=s,Ase=s/se,Apop=s*np.sqrt(20),Vse=(s/se)**2,
                                          C=float(np.mean((sublo<=0)&(subhi>=0))),Mbelow=below,Mabove=above,T=below-above,
                                          mean_width=float(np.mean(subhi-sublo)),warning_rate=float(np.mean(warning[mi,:size])))
        result.append(row)
    return result


def original_metrics(panel):
    """All J reductions, then one independent inference unit per original."""
    samples=panel["samples"];se=_original_se(samples);output=[]
    for oi,outer in enumerate(panel["outer_id"]):
        tlo,thi=panel["lower"][oi,:,0],panel["upper"][oi,:,0]
        observed=panel["valid"][oi,:,0]&np.isfinite(tlo)&np.isfinite(thi)
        if observed.any() and (not np.all(tlo[observed]==tlo[observed][0]) or not np.all(thi[observed]==thi[observed][0])):
            raise ValueError("observed t endpoints vary across measurement replicates")
        reps=[replicate_metrics(panel["lower"][oi,rep],panel["upper"][oi,rep],panel["valid"][oi,rep],panel["tail_warning"][oi,rep],se[oi])
              for rep in range(panel["lower"].shape[1])]
        for mi,method in enumerate(METHODS):
            group=[rep[mi] for rep in reps];complete=bool(len(group)==J and all(row["complete"] for row in group))
            count=1 if method=="t" else sum(row["total_inner"] for row in group)
            valid_count=int(complete) if method=="t" else sum(row["valid_inner"] for row in group)
            row=dict(distribution=_meta(panel,"distribution"),master_seed=_meta(panel,"master_seed"),outer_id=int(outer),method=method,
                     se=float(se[oi]),sample_sha256=sample_hash(samples[oi]),complete=complete,complete_replicates=sum(rep["complete"] for rep in group),
                     planned_replicates=J,total_inner=count,valid_inner=valid_count,invalid_inner=count-valid_count,replicates=group,
                     by_K=[dict.fromkeys(BASE_METRICS,None) for _ in range(2)])
            names=("d","Q30","Q120","W","ratio_Q120_Q30","mean_A30","mean_A120","mean_V30","mean_V120",
                   "mean_Jgap","mean_Pgap","mean_Rgap","mean_delta","max_abs_decomposition_residual","max_abs_lower_ss_residual","max_abs_upper_ss_residual")
            row.update(dict.fromkeys(names,None))
            row.update(dict.fromkeys(("max_abs_decomposition_normalized_residual","max_abs_lower_ss_normalized_residual","max_abs_upper_ss_normalized_residual"),None))
            if complete:
                a30=np.array([rep["A30"] for rep in group]);a120=np.array([rep["A120"] for rep in group])
                q30,q120=_variance(a30),_variance(a120)
                row.update(d=float(np.mean(a120-a30)),Q30=q30,Q120=q120,W=q120-.5*q30,ratio_Q120_Q30=q120/q30 if q30>0 else None)
                for source in ("A30","A120","V30","V120","Jgap","Pgap","Rgap","delta"):
                    row[f"mean_{source}"]=float(np.mean([rep[source] for rep in group]))
                for name in ("decomposition","lower_ss","upper_ss"):
                    values=[abs(rep[f"{name}_residual"]) for rep in group]
                    row[f"max_abs_{name}_residual"]=max(values)
                    row[f"max_abs_{name}_normalized_residual"]=max(rep[f"{name}_abs_normalized_residual"] for rep in group)
                for ki in range(2):
                    row["by_K"][ki]={metric:float(np.mean([rep["by_K"][ki][metric] for rep in group])) for metric in BASE_METRICS}
                    row["by_K"][ki]["Q"]=q30 if ki==0 else q120
            output.append(row)
    return output


def inspect_panel(panel,strict_design=True):
    errors=[]
    check=lambda condition,message:errors.append(message) if not condition else None
    samples=panel["samples"];r,n=samples.shape
    j=panel["lower"].shape[1];k=panel["lower"].shape[3]
    shape=(r,j,3,k)
    check(samples.dtype==np.dtype('float64') and panel['sample_sha256'].dtype==np.dtype('S64'),'original sample/hash dtype differs')
    check(panel['outer_id'].dtype==np.dtype('uint16') and panel['replicate_id'].dtype==np.dtype('uint8') and panel['inner_id'].dtype==np.dtype('uint8'),'ID dtype differs')
    check(panel['reason_codes'].dtype==np.dtype('uint8') and panel['reason_labels'].dtype.kind=='U' and len(panel['reason_labels'])<=256,'reason dictionary dtype/size differs')
    for field in ('lower','upper','a','z0','q_low','q_high','tau_low','tau_high'):
        check(panel[field].dtype==np.dtype('float64'),f'{field} dtype differs')
    if 'observed_streams' in panel:
        check(panel['observed_streams'].shape==(r,j,k) and panel['observed_streams'].dtype.kind=='b','observed stream shape/dtype differs')
        check(panel['observed_streams'].all(),'uncomputed required streams remain')
    elif strict_design:errors.append('observed stream mask missing')
    check(n==20 and _meta(panel,"n")==20 and _meta(panel,"alpha")==.05 and _meta(panel,"B")==4999,"fixed n/alpha/B differ")
    check(np.array_equal(panel["outer_id"],np.arange(r)),"outer IDs differ from fixed prefix")
    check(np.array_equal(panel["replicate_id"],np.arange(j)),"replicate IDs differ")
    check(np.array_equal(panel["inner_id"],np.arange(k)),"inner IDs differ")
    check(list(panel["methods"].astype(str))==list(METHODS),"method axis differs")
    if strict_design:
        check((r,j,k)==(10,20,120) and _meta(panel,"R")==10 and _meta(panel,"J")==20 and _meta(panel,"K")==120,"main R/J/K differ")
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
    for name,value in dict(distribution=dist,master_seed=seed,outer_id=0,replicate_id=0,inner_id=inner,n=20,B=4999,J=20,K=120,
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


def band_status(interval,complete=True):
    if not complete or not interval["defined"]:return "inconclusive"
    lo,hi=interval["low"],interval["high"]
    if lo>-.0025 and hi<.0025:return "supported"
    if lo>.0025 or hi<-.0025:return "not_supported"
    return "inconclusive"


def reduction_status(interval,complete=True):
    if not complete or not interval["defined"]:return "inconclusive"
    if interval["high"]<0:return "supported"
    if interval["low"]>=0:return "not_supported"
    return "inconclusive"


def joint_status(statuses):
    if "not_supported" in statuses:return "not_supported"
    return "supported" if list(statuses)==["supported","supported"] else "inconclusive"


def _expected_originals(rows,dist,seed,method,metric):
    index={row["outer_id"]:row for row in rows if row["distribution"]==dist and row["master_seed"]==seed and row["method"]==method}
    return np.array([index[outer][metric] if outer in index and index[outer]["complete"] else np.nan for outer in range(R)],dtype=float)


def original_vector(rows,dist,seed,method,metric):
    seeds=(0,1,2) if seed=="pooled" else (seed,)
    return np.concatenate([_expected_originals(rows,dist,master,method,metric) for master in seeds])


def confirmation(rows):
    quantities={}
    for dist in DISTRIBUTIONS:
        for name,metric,rule in (("H1","d",band_status),("H2","W",reduction_status)):
            values=original_vector(rows,dist,"pooled","bca",metric)
            interval=mean_interval(values)
            quantities[f"{name}_{dist}"]=dict(interval=interval,status=rule(interval),values=values)
    for name in ("H1","H2"):
        quantities[name]=dict(status=joint_status([quantities[f"{name}_{dist}"]["status"] for dist in DISTRIBUTIONS]))
    return quantities


def _interval_record(values):
    result=mean_interval(values)
    defined=result.pop('defined')
    result['valid']=defined
    result['reason']=('' if defined else 'values_nonfinite' if result['finite_count']!=result['N'] else 'too_few_originals' if result['N']<2 else 'variance_zero_or_nonfinite')
    return result


def derived_tables(panel):
    """Independent flattening: preserved original/r/K/group identities."""
    rows=original_metrics(panel);tables={name:[] for name in ('originals','original_K','replicates','groups','K_rows')}
    scalar_fields=('V30','A30','V120','A120','Vbar','Abar','Jgap','Pgap','Rgap','delta','decomposition_residual','decomposition_normalized_residual',
                   'decomposition_abs_normalized_residual','lower_ss_residual','upper_ss_residual','lower_ss_scale','upper_ss_scale',
                   'lower_ss_normalized_residual','upper_ss_normalized_residual','lower_ss_abs_normalized_residual','upper_ss_abs_normalized_residual')
    group_fields=('lower_mean','upper_mean','lower_var','upper_var','V','A')
    for row in rows:
        oi=row['outer_id'];mi=METHODS.index(row['method']);method=row['method'];r=panel['samples'].shape[0];j=panel['lower'].shape[1];k=panel['lower'].shape[3];g=k//4
        meta={key:row[key] for key in ('distribution','master_seed','outer_id','method','se','sample_sha256')}
        meta.update(n=20,R=int(_meta(panel,'R')),J=j,K_total=k,B=4999,alpha=.05)
        tables['originals'].append({**meta,**{key:value for key,value in row.items() if key not in ('replicates','by_K')}})
        for ki,limit in enumerate((g,k)):
            mask=panel['valid'][oi,:,mi,:limit]&np.isfinite(panel['lower'][oi,:,mi,:limit])&np.isfinite(panel['upper'][oi,:,mi,:limit])
            total=1 if method=='t' else j*limit;count=row['valid_inner'] if method=='t' else int(mask.sum())
            item={**meta,'K':limit,'complete':row['complete'],'total_inner':total,'valid_inner':count,'invalid_inner':total-count,**row['by_K'][ki]}
            item.update(A=row['mean_A30' if ki==0 else 'mean_A120'],V=row['mean_V30' if ki==0 else 'mean_V120'],Q=row['Q30' if ki==0 else 'Q120'])
            tables['original_K'].append(item)
        for rep_id,rep in enumerate(row['replicates']):
            reason='' if rep['complete'] else 'sample_se_invalid' if not np.isfinite(row['se']) or row['se']<=0 else 'algebraic_identity_failed' if rep['V30'] is not None else 'endpoint_invalid_or_nonfinite'
            item={**meta,'replicate_id':rep_id,**{key:rep.get(key) for key in scalar_fields},
                  **{key:rep[key] for key in ('complete','algebra_ok','total_inner','valid_inner','invalid_inner')},'reason':reason}
            item.update({gap+'_sign':int(np.sign(rep[gap])) if finite_number(rep[gap]) is not None else 'NA' for gap in ('Jgap','Pgap','Rgap','delta')})
            tables['replicates'].append(item)
            for group in range(4):
                values={key:float(rep['group_'+key][group]) for key in group_fields}
                tables['groups'].append({**meta,'replicate_id':rep_id,'group':group,'group_start':group*g,'group_stop':(group+1)*g,**values,'complete':bool(np.isfinite(list(values.values())).all())})
            for ki,limit in enumerate((g,k)):
                mask=panel['valid'][oi,rep_id,mi,:limit]&np.isfinite(panel['lower'][oi,rep_id,mi,:limit])&np.isfinite(panel['upper'][oi,rep_id,mi,:limit])
                total=1 if method=='t' else limit;count=rep['valid_inner'] if method=='t' else int(mask.sum())
                tables['K_rows'].append({**meta,'replicate_id':rep_id,'K':limit,'complete':rep['complete'],'total_inner':total,'valid_inner':count,'invalid_inner':total-count,**rep['by_K'][ki]})
    return tables


def metric_statistics(values):
    values=np.asarray(values,dtype=float);interval=mean_interval(values)
    result=dict(mean=interval['mean'],sd=interval['sd'],mcse=interval['mcse'],decision_low=interval['low'],decision_high=interval['high'])
    quantiles=np.quantile(values,[.05,.25,.5,.75,.95],method='linear') if len(values) and np.isfinite(values).all() else [None]*5
    result.update({f'p{int(q*100):02d}':finite_number(value) for q,value in zip((.05,.25,.5,.75,.95),quantiles)})
    return result


def summary_rows(original_K,pooled=False):
    output=[]
    for dist in DISTRIBUTIONS:
        for seed in (['pooled'] if pooled else (0,1,2)):
            seeds=(0,1,2) if seed=='pooled' else (seed,);planned=R*len(seeds)
            for method in METHODS:
                for kval in (30,120):
                    group=[row for row in original_K if row['distribution']==dist and row['master_seed'] in seeds and row['method']==method and row['K']==kval]
                    complete=sum(row['complete'] for row in group)
                    row=dict(distribution=dist,master_seed=seed,n=20,R=planned,J=20,K=kval,K_total=120,B=4999,alpha=.05,method=method,
                             original_total=planned,observed_original_count=len(group),missing_original_count=planned-len(group),complete_original_count=complete,
                             incomplete_original_count=planned-complete,conditional_original_count=complete,
                             interval_count=sum(item['total_inner'] for item in group),valid_count=sum(item['valid_inner'] for item in group),invalid_count=sum(item['invalid_inner'] for item in group))
                    for metric in ('A','V','Q',*BASE_METRICS):
                        values=np.concatenate([_expected_originals(group,dist,s,method,metric) for s in seeds])
                        for prefix,array in (('',values),('conditional_',values[np.isfinite(values)])):
                            for suffix,value in metric_statistics(array).items():row[prefix+metric+('' if suffix=='mean' else '_'+suffix)]=value
                    output.append(row)
    return output


def hypothesis_reference(originals):
    quantities={};seed_details=[]
    for dist in DISTRIBUTIONS:
        for H,field,rule in (('H1','d',band_status),('H2','W',reduction_status)):
            values=original_vector(originals,dist,'pooled','bca',field);interval=mean_interval(values);status=rule(interval)
            reason='' if status!='inconclusive' else ('incomplete_or_undefined_primary_comparison' if not interval['defined'] else 'margin_touched_or_crossed' if H=='H1' else 'zero_touched_or_crossed')
            quantities[f'{H}_{dist}']={**_interval_record(values),'status':status,'reason':reason,'field':field,'population':dist,'conditional_interval':_interval_record(values[np.isfinite(values)])}
    for seed in (0,1,2):
        seed_details.append(dict(master_seed=seed,**{f'{H}_{dist}':_interval_record(_expected_originals(originals,dist,seed,'bca',field)) for dist in DISTRIBUTIONS for H,field in (('H1','d'),('H2','W'))}))
    return dict(question_id='RQ-M07',confirmatory_intervals=4,decision_confidence=.99,independent_originals_per_population=30,target_method='bca',B=4999,J=20,K_first=30,K_full=120,
                seed_direction_gate=False,H1_margin=.0025,H1_units='Ase units, not percentage points',H2_variance_multiplier=.5,
                quantities=quantities,seed_details=seed_details,hypotheses={H:dict(status=joint_status([quantities[f'{H}_{d}']['status'] for d in DISTRIBUTIONS]),**{d:quantities[f'{H}_{d}']['status'] for d in DISTRIBUTIONS}) for H in ('H1','H2')})


def decomposition_rows(originals,replicates):
    output=[]
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2,'pooled'):
            seeds=(0,1,2) if seed=='pooled' else (seed,)
            for method in METHODS:
                group=[row for row in replicates if row['distribution']==dist and row['master_seed'] in seeds and row['method']==method]
                row=dict(distribution=dist,master_seed=seed,method=method,original_total=10*len(seeds),diagnostic_replicates=len(group),interpretation='r-level signs/quantiles are descriptive; original-level intervals use N10/30')
                for gap in ('Jgap','Pgap','Rgap','delta'):
                    raw=np.array([finite_number(item[gap]) if finite_number(item[gap]) is not None else np.nan for item in group]);finite=raw[np.isfinite(raw)]
                    row.update({gap+'_finite_count':len(finite),gap+'_negative_count':int((finite<0).sum()),gap+'_zero_count':int((finite==0).sum()),gap+'_positive_count':int((finite>0).sum()),gap+'_raw_mean':float(finite.mean()) if len(finite) else None})
                    quantiles=np.quantile(finite,[.05,.25,.5,.75,.95],method='linear') if len(finite) else [None]*5
                    for q,value in zip((.05,.25,.5,.75,.95),quantiles):row[f'{gap}_raw_p{int(q*100):02d}']=finite_number(value)
                    values=original_vector(originals,dist,seed,method,'mean_'+gap)
                    for prefix,array in (('',values),('conditional_',values[np.isfinite(values)])):
                        for key,value in _interval_record(array).items():row[f'{prefix}{gap}_original_{key}']=value
                for field in ('decomposition_residual','lower_ss_residual','upper_ss_residual','decomposition_normalized_residual','lower_ss_normalized_residual','upper_ss_normalized_residual'):
                    finite=[abs(float(item[field])) for item in group if finite_number(item[field]) is not None]
                    row[field+'_max_abs']=max(finite) if finite else None
                row['algebra_failed_replicates']=sum(item['reason']=='algebraic_identity_failed' for item in group)
                row['algebra_unchecked_replicates']=sum(not item['algebra_ok'] and item['reason']!='algebraic_identity_failed' for item in group)
                row['algebra_passed_replicates']=sum(bool(item['algebra_ok']) for item in group)
                ratio=original_vector(originals,dist,seed,method,'ratio_Q120_Q30')
                q30=original_vector(originals,dist,seed,method,'Q30');q120=original_vector(originals,dist,seed,method,'Q120')
                finite=np.isfinite(ratio);paired=np.isfinite(q30)&np.isfinite(q120)
                row.update(ratio_planned_originals=len(ratio),ratio_finite_originals=int(finite.sum()),ratio_missing_or_undefined_originals=int((~finite).sum()),
                           mean_individual_Q120_Q30=float(np.mean(ratio)) if finite.all() else None,
                           conditional_mean_individual_Q120_Q30=float(np.mean(ratio[finite])) if finite.any() else None,
                           ratio_mean_Q120_over_mean_Q30=float(np.mean(q120)/np.mean(q30)) if paired.all() and np.mean(q30)>0 else None,
                           conditional_ratio_mean_Q120_over_mean_Q30=float(np.mean(q120[paired])/np.mean(q30[paired])) if paired.any() and np.mean(q30[paired])>0 else None,
                           conditional_ratio_of_means_originals=int(paired.sum()),ratio_reference='descriptive only; mean of individual ratios differs from ratio of population means; zero denominator NA')
                output.append(row)
    return output


def check_inputs(manifest,panels,root=ROOT):
    """Verify accepted source bytes and every selected M06/M05 original, no RNG."""
    errors=[];inputs=manifest['inputs'];check=lambda condition,message:errors.append(message) if not condition else None
    check(inputs.get('accepted_head')==ACCEPTED_INPUT and inputs.get('source_question_id')=='RQ-M06' and inputs.get('selected_originals')==60 and inputs.get('originals_per_cell')==10 and inputs.get('n')==20,'input fixed selection metadata differs')
    for field,head in (('source_files_sha256',ACCEPTED_INPUT),('upstream_files_sha256',inputs['upstream_accepted_head'])):
        mapping=inputs[field];source_question='rq_m06' if field=='source_files_sha256' else 'rq_m05'
        expected_paths={f'evidence/{source_question}/manifest.json',*(f'evidence/{source_question}/panels/{dist}_seed{seed}.npz' for dist in DISTRIBUTIONS for seed in (0,1,2))}
        check(set(mapping)==expected_paths,f'{field}: expected exact manifest+six panel paths')
        for name,digest in mapping.items():
            path=root/name
            if path.is_absolute() and root not in path.parents:errors.append('source path outside project');continue
            data=path.read_bytes();check(hashlib.sha256(data).hexdigest()==digest,f'{name}: source file hash differs')
            committed=subprocess.check_output(['git','show',f'{head}:{name}'],cwd=root)
            check(data==committed,f'{name}: differs from accepted Git blob')
    check(inputs['upstream_accepted_head']=='dc2501c3099f5c25227d9eee00d227c2f0b02f81','upstream accepted head differs')
    selections=inputs['selections'];check(len(selections)==6 and {(item['distribution'],item['master_seed']) for item in selections}==set(panels),'input selection cells differ')
    for selection in selections:
        key=(selection['distribution'],selection['master_seed']);panel=panels.get(key)
        check(selection['source_path']==f'evidence/rq_m06/panels/{key[0]}_seed{key[1]}.npz' and selection['m05_source_path']==f'evidence/rq_m05/panels/{key[0]}_seed{key[1]}.npz','selected source paths differ from fixed panel correspondence')
        if panel is None:errors.append('selected source cell missing');continue
        with np.load(root/selection['source_path'],allow_pickle=False) as source:old={key:source[key] for key in source.files}
        with np.load(root/selection['m05_source_path'],allow_pickle=False) as source:upstream={key:source[key] for key in source.files}
        for src,size,question_namespace in ((old,200,0x4D3036),(upstream,10000,0x4D3035)):
            check(_meta(src,'n')==20 and _meta(src,'R')==size and _meta(src,'alpha')==.05 and _meta(src,'namespace')==question_namespace,'source n/R/alpha/namespace differs')
            check(src['samples'].shape==(size,20) and np.array_equal(src['outer_id'],np.arange(size)),'source shape/IDs differ')
            check(all(sample_hash(sample)==digest for sample,digest in zip(src['samples'],src['sample_sha256'].astype(str))),'source original hash differs')
        check(_meta(old,'K')==30 and np.array_equal(old['B_values'],[999,4999]) and _meta(old,'input_accepted_head')==inputs['upstream_accepted_head'],'M06 K/B/upstream metadata differs')
        check(_meta(upstream,'K')==1 and _meta(upstream,'B')==4999,'M05 K/B metadata differs')
        check(np.array_equal(old['samples'],upstream['samples'][:200]) and np.array_equal(old['sample_sha256'],upstream['sample_sha256'][:200]),'M06/M05 original provenance differs')
        check(selection['outer_ids']==list(range(10)) and np.array_equal(panel['samples'],old['samples'][:10]),'new panel is not exact fixed M06 prefix')
        check(selection['sample_sha256']==old['sample_sha256'][:10].astype(str).tolist(),'input selected row hashes differ')
        check(selection['selected_samples_sha256']==sample_hash(old['samples'][:10]),'selected matrix hash differs')
    return errors


def _read_table(path):
    import gzip
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def audit_evidence(directory,*,root=ROOT):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text());errors=[];panels={};alltables={name:[] for name in ('originals','original_K','replicates','groups','K_rows')};hashes={}
    expected_design=dict(distributions=list(DISTRIBUTIONS),n=20,master_seeds=[0,1,2],R=10,J=20,K=120,K_first=30,groups=4,B=4999,alpha=.05,
                         outer_ids=list(range(10)),methods=list(METHODS),input_accepted_head=ACCEPTED_INPUT,source_question='RQ-M06')
    errors.extend(compare_json(expected_design,manifest.get('design',{}),location='design'))
    if manifest.get('question_id')!='RQ-M07' or manifest.get('schema_version')!=1:errors.append('manifest question/schema differs')
    expected_tolerance=dict(numeric_rtol=NUMERIC_RTOL,numeric_atol=NUMERIC_ATOL,identity_rtol=IDENTITY_RTOL,identity_atol=IDENTITY_ATOL)
    errors.extend(compare_json(expected_tolerance,manifest.get('numeric_tolerances',{}),location='numeric_tolerances'))
    rng=manifest.get('randomness',{})
    errors.extend(compare_json(dict(main_namespace=MAIN_NAMESPACE,distribution_ids=dict(gamma=1,lognormal=2),spawn_key=['namespace','distribution_id','n','outer_id','purpose_id','replicate_id','inner_id'],purpose_ids=dict(bootstrap=1),replicate_ids=list(range(20)),inner_ids=list(range(120)),outer_ids=list(range(10)),native_indices_dtype='int64'),rng,location='randomness'))
    for dist in DISTRIBUTIONS:
        for seed in (0,1,2):
            path=directory/'panels'/f'{dist}_seed{seed}.npz'
            if not path.exists():errors.append(f'missing panel {dist}/{seed}');continue
            hashes[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as source:panel={key:source[key] for key in source.files}
            panels[(dist,seed)]=panel;errors.extend(f'{dist}/{seed}: {message}' for message in inspect_panel(panel))
            tables=derived_tables(panel)
            for name,rows in tables.items():alltables[name].extend(rows)
    if len(list((directory/'panels').glob('*.npz')))!=6:errors.append('saved panel count differs from six')
    errors.extend(check_inputs(manifest,panels,root=root))
    original,original_K=alltables['originals'],alltables['original_K'];reps=alltables['replicates']
    comparisons=[('original_metrics.csv',original,['distribution','master_seed','outer_id','method']),
                 ('original_K_metrics.csv',original_K,['distribution','master_seed','outer_id','method','K']),
                 ('replicate_metrics.csv.gz',reps,['distribution','master_seed','outer_id','replicate_id','method']),
                 ('group_metrics.csv.gz',alltables['groups'],['distribution','master_seed','outer_id','replicate_id','method','group']),
                 ('K_metrics.csv.gz',alltables['K_rows'],['distribution','master_seed','outer_id','replicate_id','method','K']),
                 ('summary.csv',summary_rows(original_K),['distribution','master_seed','method','K']),
                 ('pooled_summary.csv',summary_rows(original_K,True),['distribution','master_seed','method','K']),
                 ('decomposition_summary.csv',decomposition_rows(original,reps),['distribution','master_seed','method'])]
    counts={}
    for name,expected,keys in comparisons:
        observed=_read_table(directory/name);counts[name]=dict(reconstructed=len(expected),saved=len(observed));errors.extend(compare_rows(expected,observed,keys,label=name))
    errors.extend(compare_json(hypothesis_reference(original),json.loads((directory/'hypotheses.json').read_text())))
    reconstructed=0
    for (dist,seed),panel in panels.items():
        for inner in (0,29,30,119):
            path=directory/'streams'/f'{dist}_seed{seed}_outer0_replicate0_inner{inner}.npz'
            if not path.exists():errors.append(f'missing designated stream {dist}/{seed}/{inner}');continue
            hashes[str(path.relative_to(directory))]=hashlib.sha256(path.read_bytes()).hexdigest()
            errors.extend(f'{path.name}: {error}' for error in audit_stream(path,panel,dist,seed,inner));reconstructed+=1
    if len(list((directory/'streams').glob('*.npz')))!=24:errors.append('saved stream count differs from24')
    resources=json.loads((directory/'resources.json').read_text())
    errors.extend(compare_json(dict(status='completed',completed_cells=6,incomplete_cells=[]),resources,location='resources'))
    return dict(question_id='RQ-M07',status='passed' if not errors else 'failed',mismatches=errors,mismatch_count=len(errors),original_samples=sum(len(panel['samples']) for panel in panels.values()),
                table_counts=counts,fully_reconstructed_streams=reconstructed,other_generation_sequences_not_reconstructed=144000-24,
                numeric_tolerances=expected_tolerance,saved_file_sha256=hashes,
                limitations=['Only24 designated saved-index streams reconstruct bootstrap sampling, means and intervals; remaining143976 sequences are not reconstructed.',
                             'All saved endpoint metrics, identities, summaries and four original-level confirmation intervals independently reconstructed; seed/RNG metadata does not prove unrecorded streams.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path,nargs='?',default=ROOT/'evidence/rq_m07');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit_evidence(args.directory)
    if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],mismatch_count=result['mismatch_count'],table_counts=result['table_counts']),allow_nan=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
