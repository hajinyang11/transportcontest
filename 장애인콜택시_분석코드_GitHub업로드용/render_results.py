"""Render the submitted scope figure from reproduced aggregate tables."""
from pathlib import Path
from io import BytesIO
import json,argparse
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing,String,Rect,Line
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parent
A=ROOT/'analysis/calltaxi_operations_scope/multi_region_wait/annual_alternative_weather'
FIG=ROOT/'results/figures'
FONT='SubmissionMalgun';BOLD='SubmissionMalgunBold'
BLUE='#176B91';GRAY='#ADB8C3';INK='#172A38';MUTED='#526575'
def drawing(height):
    d=Drawing(1100,height);d.add(Rect(0,0,1100,height,fillColor=HexColor('#ffffff'),strokeColor=None));return d

def label(d,x,y,text,size=22,color=INK,bold=False,anchor='start'):
    d.add(String(x,y,str(text),fontName=BOLD if bold else FONT,fontSize=size,fillColor=HexColor(color),textAnchor=anchor))

def rectangle(d,x,y,w,h,color):
    d.add(Rect(x,y,w,h,fillColor=HexColor(color),strokeColor=None))

def export(d,name):
    memory=BytesIO();renderPDF.drawToFile(d,memory)
    pdf=pdfium.PdfDocument(memory.getvalue());page=pdf[0];bitmap=page.render(scale=2)
    bitmap.to_pil().save(FIG/name)
    page.close();pdf.close()

def figure3():
    source=A/'boarding_intervals/short_interval_scope'
    assert json.loads((source/'verification.json').read_text(encoding='utf-8'))['passed']
    gates=pd.read_csv(source/'gate_summary.csv')
    frozen=pd.read_csv(source/'frozen_candidate_summary.csv')
    d=drawing(880)
    label(d,48,824,'45분 이내 안내를 제공할 수 있는 조건 확인',30,bold=True)
    label(d,48,785,'2025년 하반기 연구 대상 57,291건 · 전날 예측 폭으로 적용 여부 판단',19,MUTED)
    label(d,48,722,'예측 범위가 짧을 때 안내하는 규칙',24,bold=True)
    label(d,48,683,'최대 범위 폭',18)
    label(d,360,683,'적용 비중',18)
    label(d,1045,683,'실제 포함률 / 평균 폭',18,anchor='end')
    for cap,y in [(30,612),(45,529)]:
        r=gates[gates.cap.eq(cap)&gates.phase.eq('H2')&gates.method.eq('recent_weather')].iloc[0]
        label(d,48,y+10,f'{cap}분 이하',23,bold=True)
        rectangle(d,360,y,r.share/.5*360,38,BLUE if cap==45 else GRAY)
        label(d,360+r.share/.5*360+12,y+10,f'{r.share*100:.1f}%',21,bold=True)
        label(d,1045,y+10,f'{r.coverage*100:.1f}% / {r.width_mean:.1f}분',21,bold=True,anchor='end')
    label(d,48,461,'30분 이하는 일부에 한정 · 45분 이하 규칙은 약 4할에 적용',20,bold=True)
    label(d,48,395,'상반기에 고른 15개 구·시간 조건을 고정해 하반기 확인',23,bold=True)
    r=frozen[frozen.cap.eq(45)&frozen.phase.eq('H2')&frozen.method.eq('recent_weather')&frozen.panel.eq('all_frozen_candidates')].iloc[0]
    for x,title,value in [(48,'연구 대상 중 적용 비중',f'{r.share*100:.1f}%'),
                          (390,'실제 범위 안 탑승',f'{r.coverage*100:.1f}%'),
                          (732,'평균 범위 폭',f'{r.width_mean:.1f}분')]:
        rectangle(d,x,242,310,111,'#EAF1F6')
        label(d,x+155,317,title,18,anchor='middle')
        label(d,x+155,265,value,32,BLUE,bold=True,anchor='middle')
    label(d,48,204,'8,489건 · 15개 중 12개는 하반기 100건·20일 이상 · 3개는 표본 부족',18,MUTED)
    label(d,48,159,'비중의 분모는 두 예보 격자에 연결된 47개 동의 완료 기록 · 서울 전체 비중 아님',17,MUTED)
    label(d,48,121,'서울 동일 정제 기준 완료 기록 중 확인한 45분 이하 사례는 하반기 4.4% · 나머지 지역 미평가',16,MUTED)
    label(d,48,81,'사후에 잘 맞은 하반기 조건만 재선택하지 않음 · 개인별 탑승 보장이나 일정 준수 효과 아님',16,MUTED)
    label(d,48,43,'자료: 서울시설공단 탑승내역·JMA 과거 예보 연결 · 시간 순서 확인을 거친 내부 개발 검증',16,MUTED)
    export(d,'분석결과_3_좁은안내의적용범위.png')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/malgun.ttf'))
    p.add_argument('--bold-font',type=Path,default=Path('C:/Windows/Fonts/malgunbd.ttf'))
    p.add_argument('--results-root',type=Path,default=A)
    args=p.parse_args();A=args.results_root.resolve()
    if not args.font.exists() or not args.bold_font.exists():
        p.error('Provide Korean TrueType font paths with --font and --bold-font.')
    FIG.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont(FONT,str(args.font)))
    pdfmetrics.registerFont(TTFont(BOLD,str(args.bold_font)))
    figure3()
    print('Rendered results/figures/')
