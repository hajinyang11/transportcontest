"""Audit and normalize uploaded trip records without using weather values."""
from pathlib import Path
import sys, json, hashlib, re, zipfile, argparse
from collections import Counter
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime_common import pd,np
from runtime_common import DISTRICTS
from lxml import etree as ET

ROOT=HERE.parents[1]
RAW=ROOT/'data/raw'
OUT=HERE/'multi_region_wait/multiyear_trip_only'
OUT.mkdir(exist_ok=True)
TIMES=['접수일시','예정일시','배차일시','승차일시','취소일시']
KEEP=TIMES+['출발구','출발동','이용목적','차량구분']
ALIASES={'출발지구':'출발구','출발지동':'출발동','목적지구':'목적구','목적지동':'목적동','유형':'장애유형'}
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def save(x,name):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def export(x,name):x.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def canonical(x):return re.sub(r'[·ㆍ.\s]','',re.sub(r'제(?=\d)','',str(x)))

def xlsx_chunks(path):
    """Stream values only; preserve source file and worksheets."""
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                for _,e in ET.iterparse(f,events=('end',),tag=NS+'si'):
                    strings.append(''.join(e.itertext()));e.clear()
        names=[n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml',n)]
        for name in names:
            header=None;rows=[]
            with z.open(name) as f:
                for _,e in ET.iterparse(f,events=('end',),tag=NS+'row'):
                    vals=[None]*15
                    for c in e:
                        ref=c.get('r','');letters=re.match(r'[A-Z]+',ref)
                        if not letters:continue
                        col=0
                        for char in letters.group():col=col*26+ord(char)-64
                        if col>15:continue
                        v=c.find(NS+'v');kind=c.get('t')
                        value=v.text if v is not None else None
                        if kind=='s' and value is not None:value=strings[int(value)]
                        elif kind=='inlineStr':value=''.join(c.find(NS+'is').itertext())
                        vals[col-1]=value
                    if header is None:header=[ALIASES.get(v,v) for v in vals]
                    else:rows.append(vals)
                    e.clear()
                    while e.getprevious() is not None:del e.getparent()[0]
                    if len(rows)>=100000:
                        yield pd.DataFrame(rows,columns=header)[KEEP],name
                        rows=[]
                if rows:yield pd.DataFrame(rows,columns=header)[KEEP],name

def calendar(d):
    d=d.copy()
    d['hour']=d.request.dt.hour.astype('int8')
    d['hour_float']=d.hour+d.request.dt.minute/60
    d['weekday']=d.request.dt.weekday.astype('int8')
    d['month']=d.request.dt.month.astype('int8')
    d['date']=d.request.dt.normalize()
    d['year']=d.request.dt.year.astype('int16')
    d['district_id']=pd.Categorical(d.district,categories=DISTRICTS).codes
    d['month_sin']=np.sin(2*np.pi*d.month.astype(float)/12)
    d['month_cos']=np.cos(2*np.pi*d.month.astype(float)/12)
    d['canonical']=d.dong.map(canonical)
    d['key']=d.district+'|'+d.canonical
    d['decision']=d.request-pd.Timedelta(hours=24)
    return d

def prepare_year(year):
    matches=list(RAW.glob(f'*탑승내역_{year}1231.*'));assert len(matches)==1,matches
    path=matches[0];counts=Counter();vehicles=Counter();purposes=Counter();months=Counter();sheets=Counter();parts=[]
    source_offset=0;request_min=None;request_max=None
    chunks=xlsx_chunks(path) if path.suffix=='.xlsx' else ((c,'CSV') for c in pd.read_csv(path,usecols=KEEP,dtype=str,encoding='utf-8-sig',chunksize=100000))
    for i,(c,sheet) in enumerate(chunks):
        c=c.copy();c['row_id']=np.arange(source_offset,source_offset+len(c));source_offset+=len(c)
        counts['raw_rows']+=len(c);sheets[sheet]+=len(c)
        vehicles.update(c['차량구분'].fillna('<missing>').value_counts().to_dict())
        purposes.update(c['이용목적'].fillna('<missing>').value_counts().to_dict())
        for name in TIMES:
            if path.suffix=='.xlsx':
                c[name]=pd.to_datetime(pd.to_numeric(c[name],errors='coerce'),unit='D',origin='1899-12-30',errors='coerce').dt.round('ms')
            else:c[name]=pd.to_datetime(c[name],format='mixed',errors='coerce')
        lo=c['접수일시'].min();hi=c['접수일시'].max()
        if pd.notna(lo):request_min=lo if request_min is None else min(request_min,lo)
        if pd.notna(hi):request_max=hi if request_max is None else max(request_max,hi)
        months.update(c['접수일시'].dt.strftime('%Y-%m').fillna('<missing>').value_counts().to_dict())
        counts['all_canceled']+=int(c['취소일시'].notna().sum())
        counts['all_unboarded']+=int(c['승차일시'].isna().sum())
        c=c[c['차량구분'].eq('특장차')&c['출발구'].isin(DISTRICTS)].copy()
        counts['special_seoul_requests']+=len(c)
        lead=(c['예정일시']-c['접수일시']).dt.total_seconds()/60
        c=c[~c['이용목적'].str.contains('예약',na=False)&lead.between(0,1)&c['접수일시'].dt.year.eq(year)].copy()
        counts['immediate_like_requests']+=len(c)
        counts['cancel_only']+=int((c['취소일시'].notna()&c['승차일시'].isna()).sum())
        counts['board_cancel_conflicts']+=int((c['취소일시'].notna()&c['승차일시'].notna()).sum())
        counts['neither_outcome']+=int((c['취소일시'].isna()&c['승차일시'].isna()).sum())
        wait=(c['승차일시']-c['접수일시']).dt.total_seconds()/60
        valid=c['취소일시'].isna()&c['배차일시'].ge(c['접수일시'])&c['승차일시'].ge(c['배차일시'])&wait.ge(0)
        counts['boarded_with_invalid_chronology']+=int((c['승차일시'].notna()&c['취소일시'].isna()&~valid).sum())
        c=c[valid].copy();c['wait']=wait[valid]
        counts['valid_boarded']+=len(c)
        c=c.rename(columns={'접수일시':'request','승차일시':'board','출발구':'district','출발동':'dong','이용목적':'purpose'})
        parts.append(c[['row_id','request','board','district','dong','wait','purpose']])
        print('prepared',year,'chunk',i+1,'raw',counts['raw_rows'],'valid',counts['valid_boarded'],flush=True)
    d=calendar(pd.concat(parts,ignore_index=True).sort_values(['request','row_id']).reset_index(drop=True))
    assert d.row_id.is_unique and d.wait.ge(0).all()
    d.to_pickle(OUT/f'prepared_{year}.pkl')
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    meta={'year':year,'file':path.name,'sha256':sha,'counts':dict(counts),'sheets':dict(sheets),
          'request_min':request_min,'request_max':request_max,'raw_request_months':dict(months),
          'vehicles':dict(vehicles),'purposes':dict(purposes),'valid_wait_mean':d.wait.mean(),'valid_wait_median':d.wait.median(),
          'valid_wait_p90':d.wait.quantile(.9),'valid_wait_max':d.wait.max(),'valid_wait_over360':int(d.wait.gt(360).sum())}
    save(meta,f'preparation_{year}.json')
    print('completed',year,len(d),flush=True)

def context_year(d):
    year=int(d.year.iloc[0]);assert d.year.nunique()==1
    dates=pd.date_range(f'{year}-01-01',f'{year}-12-31')
    d=d.assign(board_date=d.board.dt.normalize(),over60=d.wait.gt(60).astype(int))
    daily=d.groupby(['board_date','district','hour']).agg(n=('wait','size'),total=('wait','sum'),over60=('over60','sum'))
    idx=pd.MultiIndex.from_product([dates,DISTRICTS,range(24)],names=['date','district','hour'])
    a=daily.reindex(idx,fill_value=0).to_numpy().reshape(len(dates),len(DISTRICTS),24,3)
    end=np.maximum(np.arange(len(dates))-1,0);output=pd.DataFrame(index=idx)
    def rolling(v,days):
        c=np.concatenate([np.zeros((1,)+v.shape[1:]),np.cumsum(v,axis=0)],axis=0)
        return c[end]-c[np.maximum(end-days,0)]
    for window in [7,28,56]:
        for label,values in [('near_hour',rolling(a+np.roll(a,1,axis=2)+np.roll(a,-1,axis=2),window)),
                             ('region',rolling(a.sum(axis=2,keepdims=True),window)),
                             ('city',rolling(a.sum(axis=(1,2),keepdims=True),window))]:
            n,total,high=values[...,0],values[...,1],values[...,2]
            mean=np.divide(total,n,out=np.full_like(total,np.nan),where=n>0)
            frac=np.divide(high,n,out=np.full_like(high,np.nan),where=n>0)
            for variable,v in [('count',n),('mean',mean),('over60',frac)]:
                if label=='city' and variable=='count':continue
                output[f'{label}_{window}d_{variable}']=np.broadcast_to(v,a.shape[:-1]).reshape(-1)
    return output.reset_index()

def assemble():
    # Read metadata columns only: no weather or workday value enters this comparison.
    base=pd.read_pickle(HERE/'multi_region_wait/features.pkl')[['row_id','district','dong','grid']].copy()
    base['canonical']=base.dong.map(canonical);base['key']=base.district+'|'+base.canonical
    keys=sorted(base.key.unique());ids={k:i for i,k in enumerate(keys)}
    assert len(keys)==113 and len(keys)<255
    pieces=[];contexts=[];geoaudit=[];monthly=[]
    for year in [2023,2024,2025]:
        d=pd.read_pickle(OUT/f'prepared_{year}.pkl')
        context=context_year(d);context.to_pickle(OUT/f'context_{year}.pkl')
        if year==2025:
            old=pd.read_pickle(HERE/'improved_day_ahead/context_table.pkl')
            common=old.merge(context,on=['date','district','hour'],suffixes=('_old','_new'),validate='one_to_one')
            for col in old.columns:
                if col in ['date','district','hour']:continue
                assert np.allclose(common[col+'_old'],common[col+'_new'],equal_nan=True),col
            oldprepared=pd.read_pickle(HERE/'day_ahead_wait/prepared_records.pkl')
            assert len(d)==len(oldprepared)==1003567
            x=d[['row_id','wait','request','board']].merge(oldprepared[['row_id','wait','request','board']],on='row_id',suffixes=('_new','_old'),validate='one_to_one')
            assert np.array_equal(x.request_new,x.request_old) and np.array_equal(x.board_new,x.board_old)
            assert np.allclose(x.wait_new,x.wait_old)
        by=d.groupby(['district','canonical','key']).size().rename('n').reset_index();by['year']=year;by['in_current_keys']=by.key.isin(keys)
        geoaudit.append(by)
        if year==2025:
            d=d.merge(base[['row_id','grid']],on='row_id',validate='one_to_one')
            assert len(d)==261056
        else:d=d[d.key.isin(keys)].copy()
        d['dong_id']=d.key.map(ids).astype('int16')
        d=d.merge(context,on=['date','district','hour'],validate='many_to_one')
        pieces.append(d)
        monthly.append(d.groupby(['year','month']).wait.agg(n='size',mean='mean',median='median',p90=lambda s:s.quantile(.9)).reset_index())
        print('assembled',year,'cohort',len(d),flush=True)
    out=pd.concat(pieces,ignore_index=True).sort_values(['request','row_id']).reset_index(drop=True)
    assert not out[['year','row_id']].duplicated().any()
    out.to_pickle(OUT/'features.pkl')
    export(pd.concat(geoaudit),'historical_geography_audit.csv');export(pd.concat(monthly),'cohort_monthly.csv')
    export(out.groupby(['year','district','canonical']).size().rename('n').reset_index(),'cohort_by_dong_year.csv')
    save({'keys':keys,'records_by_year':out.year.value_counts().sort_index().to_dict(),
          'same_2025_records_and_context_verified':True,'features_weather_used':False,
          'historical_names_are_not_boundary_reconstruction':True,
          'protocol_sha256':hashlib.sha256((HERE/'MULTIYEAR_TRIP_ONLY_PROTOCOL.md').read_bytes()).hexdigest()},'assembly_audit.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int);p.add_argument('--all',action='store_true');p.add_argument('--assemble',action='store_true');a=p.parse_args()
    if a.year:prepare_year(a.year)
    if a.all:
        for year in [2023,2024,2025]:
            if not (OUT/f'prepared_{year}.pkl').exists():prepare_year(year)
    if a.assemble:assemble()
