"""ASCE-TS core algorithms. See README for manuscript/implementation distinctions."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, softmax
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

EPS = 1e-12


def dtw(x, y, tx=None, ty=None, temporal_penalty=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y):
        raise ValueError('DTW requires nonempty sequences')
    D = np.full((len(x)+1, len(y)+1), np.inf)
    D[0, 0] = 0.
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            cost = np.linalg.norm(x[i-1]-y[j-1])
            if temporal_penalty is not None:
                cost += temporal_penalty(tx[i-1], ty[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[-1, -1])


def twdtw(x, tx, y, ty, alpha=0.1, beta_days=30.):
    """Eq. 12; logistic omega is an IMPLEMENTATION DEFAULT, not specified by paper."""
    return dtw(x, y, tx, ty, lambda a,b: float(expit(alpha*(abs(float(a-b))-beta_days))))


def asynchronous_enhance(times_opt, x_opt, times_sar, x_sar,
                         before=3, after=6, analog_states=5, max_pair_gap_days=6.):
    """Eqs. 2-4; native dates retained; SAR is the output observation backbone.

    Historical optical dates are associated with nearest SAR states for computing
    SAR trajectories; association does not overwrite either acquisition date.
    max_pair_gap_days and analog_states are documented implementation defaults.
    """
    to, ts = np.asarray(times_opt, float), np.asarray(times_sar, float)
    xo, xs = np.asarray(x_opt, float), np.asarray(x_sar, float)
    if len(ts) != len(xs) or len(to) != len(xo) or not len(ts):
        raise ValueError('Inconsistent optical/SAR arrays')
    valid = np.all(np.isfinite(xo), axis=1)
    if not valid.any():
        raise ValueError('No valid optical anchors; missing optical cannot be reconstructed')
    anchors = []
    for t, v in zip(to[valid], xo[valid]):
        j = int(np.argmin(np.abs(ts-t)))
        if abs(ts[j]-t) <= max_pair_gap_days:
            anchors.append((float(t), j, v))
    if not anchors:
        raise ValueError('No SAR-supported optical anchors within max_pair_gap_days')
    def window(j):
        return xs[max(0,j-before):min(len(xs),j+after+1)]
    gaps = np.diff(np.sort(np.unique(np.r_[ts,to[valid]])))
    sigma_tau = float(np.median(gaps[gaps>0])) if np.any(gaps>0) else 1.
    output = []
    for j, t in enumerate(ts):
        direct = np.where(valid & np.isclose(to,t,atol=1e-8,rtol=0))[0]
        if len(direct):
            ohat = xo[direct[0]]
        else:
            candidates = [(dtw(window(j), window(ja)), abs(t-ta), v)
                          for ta,ja,v in anchors]
            # Paper defines sigma_d as median of historical SAR-trajectory distances;
            # calculate before optional top-k selection.
            distances = np.array([z[0] for z in candidates])
            sigma_d = max(float(np.median(distances)), EPS)
            candidates.sort(key=lambda z:z[0])
            if analog_states is not None:
                candidates = candidates[:analog_states]
            logits = np.array([-d/sigma_d - delta/max(sigma_tau,EPS)
                               for d,delta,_ in candidates])
            w = softmax(logits)
            ohat = w @ np.stack([v for _,_,v in candidates])
        output.append(np.r_[ohat, xs[j]])
    return np.asarray(output)


def fisher_weights(train_sequences, y, epsilon=1e-8):
    """Eq. 7-8; scaler is fitted exclusively on training samples."""
    X = np.asarray(train_sequences, float)
    if X.ndim == 3:
        X = X.mean(axis=1)
    X = StandardScaler().fit_transform(X)
    y = np.asarray(y)
    mean = X.mean(0)
    numerator = np.zeros(X.shape[1]); denominator = np.zeros(X.shape[1])
    for c in np.unique(y):
        z = X[y==c]
        numerator += len(z)*(z.mean(0)-mean)**2
        denominator += len(z)*z.var(0)
    return softmax(numerator/(denominator+epsilon))


def context_features(sequences, rows, cols, kf, sigma_s=1., sigma_t=1.,
                     neighborhood_radius=1):
    """Eqs. 5-10; fixed square neighborhood with self and deterministic order.

    Temporal similarity S_T is an implementation-defined normalized cosine
    similarity of complete trajectories, not explicitly defined in manuscript.
    """
    X = np.asarray(sequences, float)
    if X.ndim != 3 or not np.isfinite(X).all():
        raise ValueError('Expected finite (pixels,times,features) array')
    n, t, f = X.shape
    if len(kf) != f:
        raise ValueError('Fisher feature weight dimension mismatch')
    coords = np.c_[rows,cols].astype(float)
    tree = cKDTree(coords)
    flat = X.reshape(n,-1)
    norms = np.linalg.norm(flat,axis=1)
    q = (2*neighborhood_radius+1)**2
    out = np.zeros((n,t,q*f),float)
    for i in range(n):
        neighbors = tree.query_ball_point(coords[i], np.sqrt(2)*neighborhood_radius+1e-7)
        neighbors = [j for j in neighbors if np.max(np.abs(coords[j]-coords[i])) <= neighborhood_radius]
        neighbors.sort(key=lambda j:(coords[j,0]-coords[i,0],coords[j,1]-coords[i,1]))
        if len(neighbors)>q:
            raise ValueError('Duplicate pixel coordinates in context source')
        dist = np.linalg.norm(coords[neighbors]-coords[i],axis=1)
        ks = softmax(-dist**2/max(sigma_s**2,EPS))
        rho = (flat[neighbors] @ flat[i])/(np.maximum(norms[neighbors]*norms[i],EPS))
        kt = softmax(rho/max(sigma_t,EPS))
        for j, idx in enumerate(neighbors):
            dr = int(round(coords[idx,0]-coords[i,0]))
            dc = int(round(coords[idx,1]-coords[i,1]))
            slot = (dr+neighborhood_radius)*(2*neighborhood_radius+1)+(dc+neighborhood_radius)
            out[i,:,slot*f:(slot+1)*f] = X[idx]*(ks[j]*kt[j]*kf)[None,:]
    return out


def refine_class(X, outlier_std=2.):
    """Eq. 11; mean+2SD exclusion is an implementation default."""
    X = np.asarray(X,float)
    center = X.mean(axis=0)
    d = np.linalg.norm((X-center).reshape(len(X),-1),axis=1)
    keep = d <= d.mean()+outlier_std*d.std()
    if not keep.any():
        keep[np.argmin(d)] = True
    dr = d[keep]
    sd = max(float(dr.std()), EPS)
    w = softmax(-dr**2/(2*sd**2))
    return X[keep], w, keep


def weighted_kmedoids(D, weights, k, max_iter=100):
    """Eq. 13: weighted PAM-style medoid optimization."""
    D=np.asarray(D,float); w=np.asarray(weights,float)
    n=len(D)
    if D.shape!=(n,n) or not 1<=k<=n:
        raise ValueError('Invalid distance matrix or k')
    med=[int(np.argmin(w@D))]
    while len(med)<k:
        current=np.min(D[:,med],axis=1)
        gain=[-np.inf if j in med else np.sum(w*(current-np.minimum(current,D[:,j]))) for j in range(n)]
        med.append(int(np.argmax(gain)))
    med=np.asarray(med)
    for _ in range(max_iter):
        lab=np.argmin(D[:,med],axis=1)
        new=med.copy()
        for c in range(k):
            ix=np.where(lab==c)[0]
            if len(ix):
                new[c]=ix[np.argmin([np.sum(w[ix]*D[ix,j]) for j in ix])]
        if np.array_equal(new,med):
            break
        med=new
    return med, float(np.sum(w*np.min(D[:,med],axis=1)))


def learn_prototypes(X, times, y, k_candidates=(1,2,3,4), penalty=0.03,
                     outlier_std=2., alpha=.1, beta_days=30.):
    """Classwise TWDTW weighted k-medoids; penalized K criterion is an implementation default."""
    X=np.asarray(X,float); y=np.asarray(y); times=np.asarray(times,float)
    library=[]; summary={}
    for cls in np.unique(y):
        R,w,_=refine_class(X[y==cls],outlier_std)
        n=len(R)
        D=np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                D[i,j]=D[j,i]=twdtw(R[i],times,R[j],times,alpha,beta_days)
        options=[]
        scale=max(float(np.sum(w*np.max(D,axis=1))),EPS)
        for k in sorted(set(int(z) for z in k_candidates if 1<=z<=n)):
            med,obj=weighted_kmedoids(D,w,k)
            options.append((obj/scale+penalty*k, k, med, obj))
        if not options:
            raise ValueError(f'No valid prototype k for class {cls}')
        _,k,med,obj=min(options,key=lambda z:z[0])
        library.extend((cls, R[j].copy()) for j in med)
        summary[str(cls)]={'k':int(k),'weighted_distance':float(obj)}
    return library, summary


def prototype_features(X, times, library, alpha=.1, beta_days=30.):
    X=np.asarray(X,float)
    return np.array([[twdtw(x,times,proto,times,alpha,beta_days)
                      for _,proto in library] for x in X])


def rf_rfe(Xtr,ytr,Xv,yv,n_estimators=200,seed=42,step_fraction=.1,min_features=1):
    """Validation-guided RF-RFE; step_fraction is an implementation default."""
    active=np.arange(Xtr.shape[1]); best=(active.copy(),-np.inf)
    while len(active)>=min_features:
        rf=RandomForestClassifier(n_estimators=n_estimators,random_state=seed,n_jobs=-1,
                                  class_weight='balanced').fit(Xtr[:,active],ytr)
        score=f1_score(yv,rf.predict(Xv[:,active]),average='macro',zero_division=0)
        if score>best[1]: best=(active.copy(),float(score))
        if len(active)==min_features: break
        drop=min(len(active)-min_features,max(1,int(np.ceil(len(active)*step_fraction))))
        active=np.delete(active,np.argsort(rf.feature_importances_)[:drop])
    return best


class CascadeForest:
    """OOF cascade avoids in-sample training probability leakage."""
    def __init__(self,n_estimators=200,max_layers=5,seed=42,n_folds=3):
        self.n_estimators=n_estimators; self.max_layers=max_layers
        self.seed=seed; self.n_folds=n_folds; self.layers=[]

    def _models(self,layer):
        return [RandomForestClassifier(n_estimators=self.n_estimators,n_jobs=-1,
                    random_state=self.seed+layer,class_weight='balanced'),
                ExtraTreesClassifier(n_estimators=self.n_estimators,n_jobs=-1,
                    random_state=self.seed+100+layer,class_weight='balanced')]

    def fit(self,X,y,Xv,yv):
        base=np.asarray(X,float); basev=np.asarray(Xv,float); y=np.asarray(y)
        self.classes_=np.unique(y)
        counts=np.unique(y,return_counts=True)[1]
        folds=min(self.n_folds,int(counts.min()))
        if folds<2: raise ValueError('Each class needs >=2 training examples for OOF cascade')
        Z=base; Zv=basev; best=-np.inf; self.layers=[]
        for layer in range(self.max_layers):
            fitted=[]; oof=[]; pv=[]
            for template in self._models(layer):
                cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=self.seed+layer)
                p=np.zeros((len(y),len(self.classes_)))
                for tr,va in cv.split(Z,y):
                    from sklearn.base import clone
                    model=clone(template).fit(Z[tr],y[tr])
                    p[va]=model.predict_proba(Z[va])
                model=template.fit(Z,y)
                fitted.append(model); oof.append(p); pv.append(model.predict_proba(Zv))
            mean=np.mean(pv,axis=0)
            pred=self.classes_[mean.argmax(axis=1)]
            score=f1_score(yv,pred,average='macro',zero_division=0)
            if score<=best: break
            best=score; self.layers.append(fitted)
            Z=np.c_[base,np.concatenate(oof,axis=1)]
            Zv=np.c_[basev,np.concatenate(pv,axis=1)]
        return self

    def predict_proba(self,X):
        if not self.layers: raise RuntimeError('Cascade not fitted')
        base=np.asarray(X,float); Z=base
        for models in self.layers:
            ps=[m.predict_proba(Z) for m in models]
            p=np.mean(ps,axis=0)
            Z=np.c_[base,np.concatenate(ps,axis=1)]
        return p

    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def metrics(y,pred):
    return {'OA':float(accuracy_score(y,pred)),
            'Kappa':float(cohen_kappa_score(y,pred)),
            'F1_macro':float(f1_score(y,pred,average='macro',zero_division=0))}
