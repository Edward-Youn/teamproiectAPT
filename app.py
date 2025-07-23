import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="화성시 아파트 실거래가 분석",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 로딩 및 캐싱
@st.cache_data
def load_data():
    """데이터를 로드하고 전처리합니다."""
    try:
        df = pd.read_csv("화성_아파트_실거래_2020_2025.csv")
        
        # 데이터 전처리
        df['거래금액(만원)'] = df['거래금액(만원)'].astype(str).str.replace(',', '').astype(float)
        df['전용면적'] = pd.to_numeric(df['전용면적'], errors='coerce')
        df['계약년도'] = pd.to_numeric(df['계약년도'], errors='coerce')
        df['건축년도'] = pd.to_numeric(df['건축년도'], errors='coerce')
        
        # 평방미터당 가격 계산 (만원/㎡)
        df['평당가격'] = df['거래금액(만원)'] / df['전용면적'] * 3.3058  # 평당 가격
        df['㎡당가격'] = df['거래금액(만원)'] / df['전용면적']
        
        # 건물 연식 계산
        df['건물연식'] = df['계약년도'] - df['건축년도']
        
        # 결측값 제거
        df = df.dropna(subset=['거래금액(만원)', '전용면적', '계약년도'])
        
        return df
    except FileNotFoundError:
        st.error("CSV 파일을 찾을 수 없습니다. '화성_아파트_실거래_2020_2025.csv' 파일이 있는지 확인해주세요.")
        return None

# 편의시설 데이터 로딩
@st.cache_data
def load_facility_data():
    """체육시설과 공원 데이터를 로드합니다."""
    sports_df = None
    park_df = None
    
    # 체육시설 데이터 로드
    try:
        for encoding in ['euc-kr', 'cp949', 'utf-8-sig', 'utf-8']:
            try:
                sports_df = pd.read_csv("화성도시공사_화성시 체육시설 현황_20241231.csv", encoding=encoding)
                break
            except:
                continue
    except:
        pass
    
    # 공원 데이터 로드
    try:
        for encoding in ['euc-kr', 'cp949', 'utf-8-sig', 'utf-8']:
            try:
                park_df = pd.read_csv("화성도시공사_화성시관리공원현황_20241231.csv", encoding=encoding)
                break
            except:
                continue
    except:
        pass
    
    return sports_df, park_df

# 메인 앱
def main():
    st.title("🏠 화성시 아파트 실거래가 분석 대시보드")
    st.markdown("---")
    
    # 데이터 로드
    df = load_data()
    if df is None:
        return
    
    # 사이드바 - 필터링 옵션
    st.sidebar.header("📊 분석 옵션")
    
    # 연도 범위 선택
    year_range = st.sidebar.slider(
        "분석 연도 범위",
        min_value=int(df['계약년도'].min()),
        max_value=int(df['계약년도'].max()),
        value=(int(df['계약년도'].min()), int(df['계약년도'].max()))
    )
    
    # 전용면적 범위 선택
    area_range = st.sidebar.slider(
        "전용면적 범위 (㎡)",
        min_value=float(df['전용면적'].min()),
        max_value=float(df['전용면적'].max()),
        value=(float(df['전용면적'].min()), float(df['전용면적'].max()))
    )
    
    # 법정동 선택
    selected_dong = st.sidebar.multiselect(
        "법정동 선택",
        options=sorted(df['법정동'].unique()),
        default=sorted(df['법정동'].unique())[:5]  # 기본적으로 상위 5개만 선택
    )
    
    # 데이터 필터링
    filtered_df = df[
        (df['계약년도'] >= year_range[0]) & 
        (df['계약년도'] <= year_range[1]) &
        (df['전용면적'] >= area_range[0]) & 
        (df['전용면적'] <= area_range[1]) &
        (df['법정동'].isin(selected_dong))
    ]
    
    if filtered_df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return
    
    # 탭으로 화면 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 아파트별 가격 변동", "🗺️ 지역별 분석", "🔥 시장 활성도", "🏞️ 생활편의시설", "📊 상세 통계", "🔍 데이터 탐색"])
    
    with tab1:
        st.header("아파트별 연도별 가격 변동")
        
        # 아파트 선택
        apt_list = sorted(filtered_df['아파트이름'].unique())
        selected_apts = st.multiselect(
            "분석할 아파트를 선택하세요 (최대 10개 권장)",
            options=apt_list,
            default=apt_list[:5] if len(apt_list) >= 5 else apt_list
        )
        
        if selected_apts:
            # 아파트별 연도별 평균 가격 계산
            apt_yearly = filtered_df[filtered_df['아파트이름'].isin(selected_apts)].groupby(['아파트이름', '계약년도']).agg({
                '거래금액(만원)': 'mean',
                '㎡당가격': 'mean',
                '평당가격': 'mean'
            }).reset_index()
            
            # 가격 단위 선택
            price_unit = st.radio(
                "가격 단위 선택",
                ["거래금액(만원)", "㎡당가격", "평당가격"],
                horizontal=True
            )
            
            # 그래프 생성
            fig = px.line(
                apt_yearly, 
                x='계약년도', 
                y=price_unit,
                color='아파트이름',
                title=f'아파트별 연도별 {price_unit} 변동',
                markers=True
            )
            
            fig.update_layout(
                height=600,
                hovermode='x unified',
                xaxis_title="연도",
                yaxis_title=price_unit
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 가격 변동률 계산
            st.subheader("📈 가격 변동률 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 연평균 증가율
                growth_rates = []
                for apt in selected_apts:
                    apt_data = apt_yearly[apt_yearly['아파트이름'] == apt].sort_values('계약년도')
                    if len(apt_data) > 1:
                        first_price = apt_data[price_unit].iloc[0]
                        last_price = apt_data[price_unit].iloc[-1]
                        years = apt_data['계약년도'].iloc[-1] - apt_data['계약년도'].iloc[0]
                        if years > 0 and first_price > 0:
                            growth_rate = ((last_price / first_price) ** (1/years) - 1) * 100
                            growth_rates.append({'아파트이름': apt, '연평균증가율(%)': growth_rate})
                
                if growth_rates:
                    growth_df = pd.DataFrame(growth_rates).sort_values('연평균증가율(%)', ascending=False)
                    st.dataframe(growth_df, use_container_width=True)
            
            with col2:
                # 최근 1년 변동률
                recent_changes = []
                current_year = filtered_df['계약년도'].max()
                prev_year = current_year - 1
                
                for apt in selected_apts:
                    current_data = apt_yearly[(apt_yearly['아파트이름'] == apt) & (apt_yearly['계약년도'] == current_year)]
                    prev_data = apt_yearly[(apt_yearly['아파트이름'] == apt) & (apt_yearly['계약년도'] == prev_year)]
                    
                    if not current_data.empty and not prev_data.empty:
                        current_price = current_data[price_unit].iloc[0]
                        prev_price = prev_data[price_unit].iloc[0]
                        change_rate = ((current_price - prev_price) / prev_price) * 100
                        recent_changes.append({'아파트이름': apt, f'{prev_year}→{current_year} 변동률(%)': change_rate})
                
                if recent_changes:
                    recent_df = pd.DataFrame(recent_changes).sort_values(f'{prev_year}→{current_year} 변동률(%)', ascending=False)
                    st.dataframe(recent_df, use_container_width=True)
    
    with tab2:
        st.header("지역별 아파트 가격 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 법정동별 평균 가격
            dong_avg = filtered_df.groupby('법정동').agg({
                '거래금액(만원)': 'mean',
                '㎡당가격': 'mean',
                '평당가격': 'mean'
            }).round(1).reset_index()
            dong_avg = dong_avg.sort_values('거래금액(만원)', ascending=False)
            
            fig_dong = px.bar(
                dong_avg.head(15), 
                x='법정동', 
                y='거래금액(만원)',
                title='법정동별 평균 거래금액 (상위 15개)',
                color='거래금액(만원)',
                color_continuous_scale='Viridis'
            )
            fig_dong.update_layout(xaxis_tickangle=45, height=500)
            st.plotly_chart(fig_dong, use_container_width=True)
        
        with col2:
            # 법정동별 거래량
            dong_count = filtered_df['법정동'].value_counts().head(15)
            
            fig_count = px.bar(
                x=dong_count.index,
                y=dong_count.values,
                title='법정동별 거래량 (상위 15개)',
                labels={'x': '법정동', 'y': '거래건수'}
            )
            fig_count.update_layout(xaxis_tickangle=45, height=500)
            st.plotly_chart(fig_count, use_container_width=True)
        
        # 지역별 가격 분포 박스플롯
        st.subheader("지역별 가격 분포")
        
        # 상위 거래량 지역만 선택
        top_dongs = filtered_df['법정동'].value_counts().head(10).index.tolist()
        dong_filtered_df = filtered_df[filtered_df['법정동'].isin(top_dongs)]
        
        fig_box = px.box(
            dong_filtered_df,
            x='법정동',
            y='거래금액(만원)',
            title='법정동별 거래금액 분포 (상위 10개 지역)'
        )
        fig_box.update_layout(xaxis_tickangle=45, height=600)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.header("시장 활성도 분석")
        
        # 1. 월별/연도별 거래량 히트맵
        st.subheader("📅 월별/연도별 거래량 히트맵")
        
        # 거래량 데이터 준비
        heatmap_data = filtered_df.groupby(['계약년도', '계약월']).size().reset_index(name='거래량')
        heatmap_pivot = heatmap_data.pivot(index='계약년도', columns='계약월', values='거래량').fillna(0)
        
        # 히트맵 생성
        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="월", y="연도", color="거래량"),
            x=[f"{i}월" for i in range(1, 13)],
            y=[f"{int(year)}년" for year in heatmap_pivot.index],
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="연도별/월별 거래량 히트맵"
        )
        
        fig_heatmap.update_layout(
            height=400,
            xaxis_title="월",
            yaxis_title="연도"
        )
        
        # 각 셀에 거래량 숫자 표시
        for i, year in enumerate(heatmap_pivot.index):
            for j, month in enumerate(heatmap_pivot.columns):
                value = heatmap_pivot.iloc[i, j]
                if value > 0:
                    fig_heatmap.add_annotation(
                        x=j, y=i,
                        text=str(int(value)),
                        showarrow=False,
                        font=dict(color="white" if value > heatmap_pivot.max().max()/2 else "black", size=10)
                    )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 히트맵 인사이트
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_month = heatmap_data.groupby('계약월')['거래량'].sum().idxmax()
            st.metric("최고 거래량 월", f"{peak_month}월")
        
        with col2:
            peak_year = heatmap_data.groupby('계약년도')['거래량'].sum().idxmax()
            st.metric("최고 거래량 연도", f"{int(peak_year)}년")
        
        with col3:
            total_trades = heatmap_data['거래량'].sum()
            st.metric("총 거래량", f"{total_trades:,}건")
        
        st.markdown("---")
        
        # 2. 평수대별 선호도 분석
        st.subheader("🏠 평수대별 선호도 분석")
        
        # 평수 구간 정의 함수
        def categorize_area(area):
            if area <= 60:
                return "소형 (60㎡ 이하)"
            elif area <= 85:
                return "중형 (60-85㎡)"
            elif area <= 135:
                return "대형 (85-135㎡)"
            else:
                return "초대형 (135㎡ 초과)"
        
        # 평수 구간 컬럼 추가
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['평수구간'] = filtered_df_copy['전용면적'].apply(categorize_area)
        
        # 연도별 평수대별 거래 비중
        yearly_area_dist = filtered_df_copy.groupby(['계약년도', '평수구간']).size().reset_index(name='거래량')
        yearly_area_pct = yearly_area_dist.groupby('계약년도').apply(
            lambda x: x.assign(비중=x['거래량'] / x['거래량'].sum() * 100)
        ).reset_index(drop=True)
        
        # 스택 바차트로 연도별 평수대 비중 표시
        fig_area_yearly = px.bar(
            yearly_area_pct,
            x='계약년도',
            y='비중',
            color='평수구간',
            title='연도별 평수대별 거래 비중 (%)',
            labels={'비중': '거래 비중 (%)', '계약년도': '연도'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_area_yearly.update_layout(
            height=500,
            xaxis_title="연도",
            yaxis_title="거래 비중 (%)",
            legend_title="평수 구간"
        )
        
        st.plotly_chart(fig_area_yearly, use_container_width=True)
        
        # 월별 평수대별 선호도
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("월별 평수대별 거래량")
            monthly_area_dist = filtered_df_copy.groupby(['계약월', '평수구간']).size().reset_index(name='거래량')
            
            fig_area_monthly = px.bar(
                monthly_area_dist,
                x='계약월',
                y='거래량',
                color='평수구간',
                title='월별 평수대별 거래량',
                labels={'거래량': '거래량 (건)', '계약월': '월'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_area_monthly.update_layout(height=400)
            st.plotly_chart(fig_area_monthly, use_container_width=True)
        
        with col2:
            st.subheader("평수대별 평균 거래금액")
            area_price = filtered_df_copy.groupby('평수구간').agg({
                '거래금액(만원)': 'mean',
                '㎡당가격': 'mean'
            }).round(0).reset_index()
            
            fig_area_price = px.bar(
                area_price,
                x='평수구간',
                y='거래금액(만원)',
                title='평수대별 평균 거래금액',
                labels={'거래금액(만원)': '평균 거래금액 (만원)'},
                color='거래금액(만원)',
                color_continuous_scale='Blues'
            )
            
            fig_area_price.update_layout(height=400, xaxis_tickangle=45)
            st.plotly_chart(fig_area_price, use_container_width=True)
        
        # 평수대별 상세 통계
        st.subheader("📊 평수대별 상세 통계")
        
        # 거래량 계산
        area_counts = filtered_df_copy.groupby('평수구간').size().reset_index(name='거래량')
        
        # 기타 통계 계산
        area_stats = filtered_df_copy.groupby('평수구간').agg({
            '거래금액(만원)': ['mean', 'median', 'std'],
            '㎡당가격': ['mean', 'median'],
            '건물연식': 'mean'
        }).round(1)
        
        # 컬럼명 정리
        area_stats.columns = ['평균금액', '중간금액', '금액표준편차', '평균㎡당가격', '중간㎡당가격', '평균건물연식']
        area_stats = area_stats.reset_index()
        
        # 거래량과 합치기
        area_stats = area_stats.merge(area_counts, on='평수구간')
        
        # 비중 계산
        area_stats['거래비중(%)'] = (area_stats['거래량'] / area_stats['거래량'].sum() * 100).round(1)
        
        # 컬럼 순서 재정렬
        area_stats = area_stats[['평수구간', '거래량', '거래비중(%)', '평균금액', '중간금액', 
                                '금액표준편차', '평균㎡당가격', '중간㎡당가격', '평균건물연식']]
        
        # 표 스타일링
        st.dataframe(
            area_stats.style.format({
                '거래량': '{:,}',
                '거래비중(%)': '{:.1f}',
                '평균금액': '{:,.0f}',
                '중간금액': '{:,.0f}',
                '금액표준편차': '{:,.0f}',
                '평균㎡당가격': '{:,.0f}',
                '중간㎡당가격': '{:,.0f}',
                '평균건물연식': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # 트렌드 분석
        st.subheader("📈 평수대별 트렌드 분석")
        
        # 최근 3년간 평수대별 선호도 변화
        recent_years = sorted(filtered_df_copy['계약년도'].unique())[-3:]
        trend_data = []
        
        for year in recent_years:
            year_data = filtered_df_copy[filtered_df_copy['계약년도'] == year]
            area_dist = year_data['평수구간'].value_counts(normalize=True) * 100
            for area_type, pct in area_dist.items():
                trend_data.append({
                    '연도': f"{int(year)}년",
                    '평수구간': area_type,
                    '비중': pct
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        if not trend_df.empty:
            fig_trend = px.line(
                trend_df,
                x='연도',
                y='비중',
                color='평수구간',
                title='최근 3년간 평수대별 선호도 변화',
                markers=True,
                labels={'비중': '거래 비중 (%)', '연도': '연도'}
            )
            
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab4:
        st.header("생활편의시설과 아파트 가격 분석")
        
        # 편의시설 데이터 로드
        sports_df, park_df = load_facility_data()
        
        if sports_df is not None and park_df is not None:
            st.success("✅ 편의시설 데이터 로드 성공")
            
            try:
                # 컬럼명 정리 (9개, 10개 컬럼 구조에 맞게)
                if len(sports_df.columns) == 9:
                    sports_df.columns = ['시설명', '공원명', '위치', '면수', '시설면적', '총면적', '관리기관', '관리자연락처', '데이터기준일자']
                
                if len(park_df.columns) == 10:
                    park_df.columns = ['개소', '공원종류', '권역', '공원명', '위치', '조성연도', '조성면적', '관리기관', '관리자연락처', '데이터기준일자']
                
                # 간단한 동 정보 추출 (첫 번째 단어 또는 읍/면/동 끝나는 부분)
                def extract_simple_dong(location):
                    if pd.isna(location):
                        return ""
                    location = str(location).strip()
                    parts = location.split()
                    if parts:
                        first_part = parts[0]
                        # 읍, 면, 동으로 끝나는지 확인
                        for suffix in ['읍', '면', '동']:
                            if first_part.endswith(suffix):
                                return first_part
                        return first_part
                    return ""
                
                sports_df['추출동'] = sports_df['위치'].apply(extract_simple_dong)
                park_df['추출동'] = park_df['위치'].apply(extract_simple_dong)
                
                # 동별 집계
                sports_summary = sports_df.groupby('추출동').agg({
                    '시설명': 'count',
                    '총면적': 'sum'
                }).rename(columns={'시설명': '체육시설수', '총면적': '체육시설면적'}).reset_index()
                
                park_summary = park_df.groupby('추출동').agg({
                    '공원명': 'count', 
                    '조성면적': 'sum'
                }).rename(columns={'공원명': '공원수', '조성면적': '공원면적'}).reset_index()
                
                # 아파트 데이터 법정동별 집계
                apt_summary = filtered_df.groupby('법정동').agg({
                    '거래금액(만원)': 'mean',
                    '㎡당가격': 'mean',
                    '아파트이름': 'count'
                }).rename(columns={'아파트이름': '거래건수'}).reset_index()
                
                # 기본 정보 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("체육시설 총 개수", f"{len(sports_df)}개")
                with col2:
                    st.metric("공원 총 개수", f"{len(park_df)}개")  
                with col3:
                    st.metric("분석 대상 법정동", f"{len(apt_summary)}개")
                
                # 데이터 매칭 (간단한 방식)
                # 체육시설과 아파트 데이터 매칭
                merged_sports = pd.merge(apt_summary, sports_summary, left_on='법정동', right_on='추출동', how='left')
                merged_sports['체육시설수'] = merged_sports['체육시설수'].fillna(0)
                merged_sports['체육시설면적'] = merged_sports['체육시설면적'].fillna(0)
                
                # 공원과 아파트 데이터 매칭  
                merged_parks = pd.merge(apt_summary, park_summary, left_on='법정동', right_on='추출동', how='left')
                merged_parks['공원수'] = merged_parks['공원수'].fillna(0)
                merged_parks['공원면적'] = merged_parks['공원면적'].fillna(0)
                
                st.markdown("---")
                
                with col1:
                    st.subheader("🏃‍♂️ 체육시설과 아파트 가격")
                    
                    # 체육시설이 있는 지역만 필터링
                    sports_data = merged_sports[merged_sports['체육시설수'] > 0]
                    
                    if len(sports_data) > 0:
                        fig_sports = px.scatter(
                            sports_data,
                            x='체육시설수',
                            y='거래금액(만원)',
                            size='거래건수',
                            hover_data=['법정동'],
                            title='체육시설 수 vs 평균 아파트 가격',
                            labels={'체육시설수': '체육시설 개수', '거래금액(만원)': '평균 거래금액 (만원)'}
                        )
                        fig_sports.update_layout(height=400)
                        st.plotly_chart(fig_sports, use_container_width=True)
                        
                        # 상관계수
                        if len(sports_data) > 1:
                            corr = sports_data['체육시설수'].corr(sports_data['거래금액(만원)'])
                            st.metric("상관계수", f"{corr:.3f}")
                    else:
                        st.info("체육시설 데이터와 매칭된 지역이 없습니다.")
                
                with col2:
                    st.subheader("🌳 공원과 아파트 가격")
                    
                    # 공원이 있는 지역만 필터링
                    park_data = merged_parks[merged_parks['공원수'] > 0]
                    
                    if len(park_data) > 0:
                        fig_parks = px.scatter(
                            park_data,
                            x='공원수',
                            y='거래금액(만원)',
                            size='거래건수',
                            hover_data=['법정동'],
                            title='공원 수 vs 평균 아파트 가격',
                            labels={'공원수': '공원 개수', '거래금액(만원)': '평균 거래금액 (만원)'}
                        )
                        fig_parks.update_layout(height=400)
                        st.plotly_chart(fig_parks, use_container_width=True)
                        
                        # 상관계수
                        if len(park_data) > 1:
                            corr = park_data['공원수'].corr(park_data['거래금액(만원)'])
                            st.metric("상관계수", f"{corr:.3f}")
                    else:
                        st.info("공원 데이터와 매칭된 지역이 없습니다.")
                
                # 편의시설 순위
                st.markdown("---")
                st.subheader("📊 편의시설 현황")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**체육시설 보유 상위 지역**")
                    top_sports = sports_summary.nlargest(10, '체육시설수')[['추출동', '체육시설수', '체육시설면적']]
                    top_sports['체육시설면적'] = top_sports['체육시설면적'].round(0)
                    st.dataframe(top_sports, use_container_width=True)
                
                with col2:
                    st.write("**공원 보유 상위 지역**")
                    top_parks = park_summary.nlargest(10, '공원수')[['추출동', '공원수', '공원면적']]
                    top_parks['공원면적'] = top_parks['공원면적'].round(0)
                    st.dataframe(top_parks, use_container_width=True)
                
                # 데이터 매칭 상황 확인
                with st.expander("🔍 데이터 매칭 상세 정보"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**체육시설 데이터 추출 결과**")
                        st.dataframe(sports_df[['시설명', '위치', '추출동']].head(10), use_container_width=True)
                        
                        st.write("**아파트 법정동 vs 체육시설 매칭**")
                        match_result = merged_sports[['법정동', '체육시설수']].head(10)
                        st.dataframe(match_result, use_container_width=True)
                    
                    with col2:
                        st.write("**공원 데이터 추출 결과**")
                        st.dataframe(park_df[['공원명', '위치', '추출동']].head(10), use_container_width=True)
                        
                        st.write("**아파트 법정동 vs 공원 매칭**")
                        match_result = merged_parks[['법정동', '공원수']].head(10)
                        st.dataframe(match_result, use_container_width=True)
                
            except Exception as e:
                st.error(f"편의시설 데이터 분석 중 오류 발생: {str(e)}")
                st.info("데이터 구조를 확인하고 다시 시도해주세요.")
        
        else:
            st.warning("⚠️ 편의시설 데이터를 불러올 수 없습니다.")
            st.info("""
            **필요한 파일:**
            - 화성도시공사_화성시 체육시설 현황_20241231.csv
            - 화성도시공사_화성시관리공원현황_20241231.csv
            
            파일이 같은 폴더에 있는지 확인해주세요.
            """)
    
    with tab5:
        st.header("상세 통계 분석")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 거래건수", f"{len(filtered_df):,}건")
        with col2:
            st.metric("평균 거래금액", f"{filtered_df['거래금액(만원)'].mean():,.0f}만원")
        with col3:
            st.metric("평균 전용면적", f"{filtered_df['전용면적'].mean():.1f}㎡")
        with col4:
            st.metric("평균 ㎡당 가격", f"{filtered_df['㎡당가격'].mean():.0f}만원")
        
        st.markdown("---")
        
        # 상관관계 분석
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("전용면적 vs 거래금액")
            fig_scatter = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df))),  # 성능을 위해 샘플링
                x='전용면적',
                y='거래금액(만원)',
                color='법정동',
                title='전용면적과 거래금액의 관계',
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("건물연식 vs 거래금액")
            fig_age = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df))),
                x='건물연식',
                y='거래금액(만원)',
                color='법정동',
                title='건물연식과 거래금액의 관계',
                opacity=0.7
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # 월별 거래 패턴
        st.subheader("월별 거래 패턴")
        monthly_pattern = filtered_df.groupby('계약월').agg({
            '거래금액(만원)': ['count', 'mean']
        }).round(1)
        monthly_pattern.columns = ['거래건수', '평균금액']
        monthly_pattern = monthly_pattern.reset_index()
        
        fig_monthly = make_subplots(
            rows=1, cols=2,
            subplot_titles=('월별 거래건수', '월별 평균금액'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_monthly.add_trace(
            go.Bar(x=monthly_pattern['계약월'], y=monthly_pattern['거래건수'], name='거래건수'),
            row=1, col=1
        )
        
        fig_monthly.add_trace(
            go.Scatter(x=monthly_pattern['계약월'], y=monthly_pattern['평균금액'], 
                      mode='lines+markers', name='평균금액'),
            row=1, col=2
        )
        
        fig_monthly.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab6:
        st.header("데이터 탐색")
        
        # 데이터 요약 정보
        st.subheader("📋 데이터 요약")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**기본 정보**")
            st.write(f"- 데이터 기간: {df['계약년도'].min():.0f}년 ~ {df['계약년도'].max():.0f}년")
            st.write(f"- 총 거래건수: {len(df):,}건")
            st.write(f"- 아파트 수: {df['아파트이름'].nunique():,}개")
            st.write(f"- 법정동 수: {df['법정동'].nunique()}개")
        
        with col2:
            st.write("**가격 정보**")
            st.write(f"- 최고 거래가: {df['거래금액(만원)'].max():,.0f}만원")
            st.write(f"- 최저 거래가: {df['거래금액(만원)'].min():,.0f}만원")
            st.write(f"- 중앙값: {df['거래금액(만원)'].median():,.0f}만원")
            st.write(f"- 표준편차: {df['거래금액(만원)'].std():,.0f}만원")
        
        # 상위/하위 아파트
        st.subheader("🏆 가격 순위")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**평균 거래가 상위 10개 아파트**")
            top_apts = filtered_df.groupby('아파트이름')['거래금액(만원)'].mean().sort_values(ascending=False).head(10)
            for i, (apt, price) in enumerate(top_apts.items(), 1):
                st.write(f"{i}. {apt}: {price:,.0f}만원")
        
        with col2:
            st.write("**거래량 상위 10개 아파트**")
            volume_apts = filtered_df['아파트이름'].value_counts().head(10)
            for i, (apt, count) in enumerate(volume_apts.items(), 1):
                st.write(f"{i}. {apt}: {count}건")
        
        # 원본 데이터 테이블
        st.subheader("📄 필터링된 데이터")
        st.write(f"표시 중인 데이터: {len(filtered_df):,}건")
        
        # 데이터 다운로드 버튼
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="필터링된 데이터 CSV 다운로드",
            data=csv,
            file_name=f'화성_아파트_실거래_필터링_{year_range[0]}_{year_range[1]}.csv',
            mime='text/csv'
        )
        
        # 데이터 테이블 (페이지네이션)
        st.dataframe(
            filtered_df[['법정동', '아파트이름', '전용면적', '계약년도', '계약월', 
                        '거래금액(만원)', '㎡당가격', '건축년도']].head(1000),
            use_container_width=True
        )
        
        if len(filtered_df) > 1000:
            st.info("성능상의 이유로 상위 1000건만 표시됩니다. 전체 데이터는 CSV 다운로드를 이용해주세요.")

if __name__ == "__main__":
    main()