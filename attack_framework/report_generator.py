"""
공격 결과 보고서 생성
통합 공격 결과를 JSON 및 HTML 형태로 보고서 생성
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .attack_executor import CombinedAttackResult, AttackStrategy
from .attack_templates import TemplateAttackResult
from .feedback_loop import FailureAnalysis


class ReportGenerator:
    """공격 결과 보고서 생성기"""
    
    def __init__(self):
        self.reports_dir = Path("./reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    async def generate_full_report(self, 
                                 result: CombinedAttackResult,
                                 failure_analyses: List[FailureAnalysis] = None,
                                 metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """전체 보고서 생성 (JSON + HTML)"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
            
        report_data = self._prepare_report_data(result, failure_analyses, metadata, timestamp)
        
        # JSON 보고서 생성
        json_file = await self._generate_json_report(report_data, timestamp)
        
        # HTML 보고서 생성
        html_file = await self._generate_html_report(report_data, timestamp)
        
        print(f"\n📄 보고서 생성 완료:")
        print(f"  JSON: {json_file}")
        print(f"  HTML: {html_file}")
        
        return {
            "json": json_file,
            "html": html_file,
            "timestamp": timestamp
        }
        
    def _prepare_report_data(self, 
                           result: CombinedAttackResult,
                           failure_analyses: List[FailureAnalysis],
                           metadata: Dict[str, Any],
                           timestamp: str) -> Dict[str, Any]:
        """보고서 데이터 준비"""
        
        # 기본 메타데이터
        report_metadata = {
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "strategy": result.strategy.value,
            "framework_version": "1.0.0",
            **metadata
        }
        
        # 전체 통계
        overall_stats = {
            "total_attempts": result.total_attempts,
            "total_success": result.total_success,
            "success_rate": result.success_rate,
            "execution_time": result.execution_time,
            "enhanced_attacks": result.enhanced_attacks
        }
        
        # 템플릿 공격 결과
        template_data = {
            "total_templates": len(result.template_results),
            "successful_templates": sum(1 for r in result.template_results if r.success),
            "failed_templates": sum(1 for r in result.template_results if not r.success),
            "results": []
        }
        
        for template_result in result.template_results:
            template_data["results"].append({
                "template_id": template_result.template_id,
                "prompt": template_result.template_prompt,
                "response": template_result.response,
                "success": template_result.success,
                "indicators_found": template_result.indicators_found,
                "execution_time": template_result.execution_time,
                "category": template_result.category.value,
                "cvss_score": template_result.cvss_score,
                "enhanced_with_thinking": getattr(template_result, 'enhanced_with_thinking', False)
            })
            
        # LLM-to-LLM 공격 결과
        llm_to_llm_data = {
            "total_iterations": len(result.llm_to_llm_results),
            "successful_iterations": sum(1 for r in result.llm_to_llm_results if r.success),
            "results": []
        }
        
        for llm_result in result.llm_to_llm_results:
            llm_to_llm_data["results"].append({
                "phase": llm_result.phase.value,
                "prompt": llm_result.prompt,
                "response": llm_result.response,
                "success": llm_result.success,
                "indicators_found": llm_result.indicators_found,
                "execution_time": llm_result.execution_time,
                "cvss_score": llm_result.cvss_score
            })
            
        # 실패 분석 데이터
        failure_data = None
        if failure_analyses:
            failure_data = {
                "total_analyzed": len(failure_analyses),
                "analyses": []
            }
            
            for analysis in failure_analyses:
                failure_data["analyses"].append({
                    "template_id": analysis.template_id,
                    "failure_reason": analysis.failure_reason.value,
                    "confidence": analysis.confidence,
                    "evidence_keywords": analysis.evidence_keywords,
                    "improvement_suggestions": analysis.improvement_suggestions,
                    "recommended_approach": analysis.recommended_approach
                })
                
        # CVSS 점수 분석
        cvss_analysis = self._analyze_cvss_scores(result)
        
        return {
            "metadata": report_metadata,
            "overall_statistics": overall_stats,
            "template_attacks": template_data,
            "llm_to_llm_attacks": llm_to_llm_data,
            "failure_analysis": failure_data,
            "cvss_analysis": cvss_analysis,
            "summary": self._generate_summary(result, failure_analyses)
        }
        
    async def _generate_json_report(self, report_data: Dict[str, Any], timestamp: str) -> str:
        """JSON 보고서 생성"""
        
        filename = f"report_{timestamp}.json"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        return str(filepath)
        
    async def _generate_html_report(self, report_data: Dict[str, Any], timestamp: str) -> str:
        """HTML 보고서 생성"""
        
        filename = f"report_{timestamp}.html"
        filepath = self.reports_dir / filename
        
        html_content = self._create_html_template(report_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(filepath)
        
    def _create_html_template(self, data: Dict[str, Any]) -> str:
        """HTML 보고서 템플릿 생성"""
        
        metadata = data["metadata"]
        stats = data["overall_statistics"]
        template_data = data["template_attacks"]
        llm_data = data["llm_to_llm_attacks"]
        failure_data = data.get("failure_analysis")
        cvss_data = data["cvss_analysis"]
        summary = data["summary"]
        
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP 프롬프트 주입 공격 보고서 - {metadata['timestamp']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007acc;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .success {{
            color: #28a745;
        }}
        .failure {{
            color: #dc3545;
        }}
        .warning {{
            color: #ffc107;
        }}
        .attack-result {{
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ddd;
        }}
        .attack-success {{
            background: #d4edda;
            border-left-color: #28a745;
        }}
        .attack-failure {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        .prompt-text {{
            background: #f1f3f4;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            margin: 10px 0;
            max-height: 200px;
            overflow-y: auto;
        }}
        .response-text {{
            background: #e8f4f8;
            padding: 10px;
            border-radius: 4px;
            font-size: 0.9em;
            margin: 10px 0;
            max-height: 150px;
            overflow-y: auto;
        }}
        .indicators {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
        }}
        .indicator {{
            background: #007acc;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
        }}
        .failure-analysis {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .cvss-score {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
        }}
        .cvss-low {{ background: #28a745; }}
        .cvss-medium {{ background: #ffc107; color: black; }}
        .cvss-high {{ background: #fd7e14; }}
        .cvss-critical {{ background: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .expandable {{
            cursor: pointer;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .expandable:hover {{
            background: #e9ecef;
        }}
        .content {{
            display: none;
            padding: 10px;
            border-left: 3px solid #007acc;
            margin-left: 20px;
        }}
        .content.show {{
            display: block;
        }}
    </style>
    <script>
        function toggleContent(id) {{
            const content = document.getElementById(id);
            content.classList.toggle('show');
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>🎯 MCP 기반 프롬프트 주입 공격 보고서</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['total_attempts']}</div>
                <div class="stat-label">총 공격 시도</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{stats['total_success']}</div>
                <div class="stat-label">성공한 공격</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['success_rate']:.1f}%</div>
                <div class="stat-label">전체 성공률</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['execution_time']:.1f}초</div>
                <div class="stat-label">실행 시간</div>
            </div>
        </div>
        
        <h2>📊 요약</h2>
        <div class="prompt-text">{summary}</div>
        
        <h2>🎯 템플릿 기반 공격 결과</h2>
        <p><strong>성공:</strong> <span class="success">{template_data['successful_templates']}</span> / 
           <strong>실패:</strong> <span class="failure">{template_data['failed_templates']}</span> / 
           <strong>전체:</strong> {template_data['total_templates']}</p>
"""

        # 템플릿 공격 결과 상세
        for i, result in enumerate(template_data['results']):
            status_class = "attack-success" if result['success'] else "attack-failure"
            status_text = "✅ 성공" if result['success'] else "❌ 실패"
            
            # Sequential Thinking 개선 여부 표시
            enhancement_badge = ""
            if result.get('enhanced_with_thinking', False):
                enhancement_badge = " <span style='background: #17a2b8; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;'>ST 개선</span>"
            
            cvss_class = ""
            if result['cvss_score']:
                if result['cvss_score'] >= 9.0:
                    cvss_class = "cvss-critical"
                elif result['cvss_score'] >= 7.0:
                    cvss_class = "cvss-high"
                elif result['cvss_score'] >= 4.0:
                    cvss_class = "cvss-medium"
                else:
                    cvss_class = "cvss-low"
            
            html += f"""
        <div class="attack-result {status_class}">
            <h4>템플릿 #{result['template_id']} - {status_text}{enhancement_badge}</h4>
            <div class="expandable" onclick="toggleContent('template_{i}')">
                📝 프롬프트 보기 (클릭)
            </div>
            <div id="template_{i}" class="content">
                <div class="prompt-text">{result['prompt']}</div>
            </div>
            
            <div class="expandable" onclick="toggleContent('response_{i}')">
                📄 응답 보기 (클릭)
            </div>
            <div id="response_{i}" class="content">
                <div class="response-text">{result['response']}</div>
            </div>
            
            {f'<div class="indicators">지표: ' + ''.join([f'<span class="indicator">{ind}</span>' for ind in result['indicators_found']]) + '</div>' if result['indicators_found'] else ''}
            
            <p><strong>카테고리:</strong> {result['category']} | 
               <strong>실행시간:</strong> {result['execution_time']:.2f}초
               {f' | <strong>CVSS:</strong> <span class="cvss-score {cvss_class}">{result["cvss_score"]:.1f}</span>' if result['cvss_score'] else ''}</p>
        </div>
"""

        # LLM-to-LLM 공격 결과
        if llm_data['results']:
            html += f"""
        <h2>🤖 LLM-to-LLM 보완 공격 결과</h2>
        <p><strong>성공:</strong> <span class="success">{llm_data['successful_iterations']}</span> / 
           <strong>전체:</strong> {llm_data['total_iterations']}</p>
"""
            
            for i, result in enumerate(llm_data['results']):
                status_class = "attack-success" if result['success'] else "attack-failure"
                status_text = "✅ 성공" if result['success'] else "❌ 실패"
                
                html += f"""
        <div class="attack-result {status_class}">
            <h4>{result['phase']} 단계 - {status_text}</h4>
            <div class="expandable" onclick="toggleContent('llm_prompt_{i}')">
                📝 생성된 프롬프트 보기 (클릭)
            </div>
            <div id="llm_prompt_{i}" class="content">
                <div class="prompt-text">{result['prompt']}</div>
            </div>
            
            <div class="expandable" onclick="toggleContent('llm_response_{i}')">
                📄 응답 보기 (클릭)
            </div>
            <div id="llm_response_{i}" class="content">
                <div class="response-text">{result['response']}</div>
            </div>
            
            <p><strong>실행시간:</strong> {result['execution_time']:.2f}초
               {f' | <strong>CVSS:</strong> {result["cvss_score"]:.1f}' if result['cvss_score'] else ''}</p>
        </div>
"""

        # 실패 분석
        if failure_data:
            html += f"""
        <h2>🔍 실패 분석</h2>
        <p>총 {failure_data['total_analyzed']}개의 실패 케이스 분석</p>
"""
            
            for analysis in failure_data['analyses']:
                html += f"""
        <div class="failure-analysis">
            <h4>템플릿 #{analysis['template_id']} 실패 분석</h4>
            <p><strong>실패 원인:</strong> {analysis['failure_reason']} (신뢰도: {analysis['confidence']:.2f})</p>
            <p><strong>증거 키워드:</strong> {', '.join(analysis['evidence_keywords'])}</p>
            <p><strong>권장 접근법:</strong> {analysis['recommended_approach']}</p>
            <div class="expandable" onclick="toggleContent('suggestions_{analysis["template_id"]}')">
                💡 개선 제안 보기 (클릭)
            </div>
            <div id="suggestions_{analysis['template_id']}" class="content">
                <ul>
                    {''.join([f'<li>{suggestion}</li>' for suggestion in analysis['improvement_suggestions']])}
                </ul>
            </div>
        </div>
"""

        # CVSS 분석
        html += f"""
        <h2>🛡️ CVSS 보안 점수 분석</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{cvss_data['average_score']:.1f}</div>
                <div class="stat-label">평균 CVSS 점수</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cvss_data['max_score']:.1f}</div>
                <div class="stat-label">최고 CVSS 점수</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cvss_data['critical_count']}</div>
                <div class="stat-label">치명적 취약점</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cvss_data['high_count']}</div>
                <div class="stat-label">높은 위험도</div>
            </div>
        </div>
        
        <h2>📋 메타데이터</h2>
        <table>
            <tr><th>항목</th><th>값</th></tr>
            <tr><td>생성 시간</td><td>{metadata['generated_at']}</td></tr>
            <tr><td>공격 전략</td><td>{metadata['strategy']}</td></tr>
            <tr><td>프레임워크 버전</td><td>{metadata['framework_version']}</td></tr>
        </table>
        
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
            <p>MCP 기반 프롬프트 주입 공격 프레임워크 - 보안 연구 목적으로만 사용</p>
        </footer>
    </div>
</body>
</html>"""

        return html
        
    def _analyze_cvss_scores(self, result: CombinedAttackResult) -> Dict[str, Any]:
        """CVSS 점수 분석"""
        
        all_scores = []
        
        # 템플릿 공격 점수 수집
        for template_result in result.template_results:
            if template_result.cvss_score:
                all_scores.append(template_result.cvss_score)
                
        # LLM-to-LLM 공격 점수 수집
        for llm_result in result.llm_to_llm_results:
            if llm_result.cvss_score:
                all_scores.append(llm_result.cvss_score)
                
        if not all_scores:
            return {
                "average_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0
            }
            
        return {
            "average_score": sum(all_scores) / len(all_scores),
            "max_score": max(all_scores),
            "min_score": min(all_scores),
            "critical_count": sum(1 for s in all_scores if s >= 9.0),
            "high_count": sum(1 for s in all_scores if 7.0 <= s < 9.0),
            "medium_count": sum(1 for s in all_scores if 4.0 <= s < 7.0),
            "low_count": sum(1 for s in all_scores if 0.1 <= s < 4.0)
        }
        
    def _generate_summary(self, result: CombinedAttackResult, failure_analyses: List[FailureAnalysis]) -> str:
        """공격 결과 요약 생성"""
        
        strategy_desc = {
            AttackStrategy.TEMPLATE_ONLY: "템플릿 기반 공격만 사용",
            AttackStrategy.LLM_TO_LLM_ONLY: "LLM-to-LLM 공격만 사용", 
            AttackStrategy.HYBRID: "하이브리드 공격 (템플릿 + LLM-to-LLM)"
        }
        
        summary = f"""
공격 전략: {strategy_desc.get(result.strategy, result.strategy.value)}

전체 결과:
- 총 {result.total_attempts}번의 공격 시도 중 {result.total_success}번 성공 (성공률: {result.success_rate:.1f}%)
- 실행 시간: {result.execution_time:.1f}초

단계별 결과:
- 템플릿 공격: {len(result.template_results)}번 시도, {sum(1 for r in result.template_results if r.success)}번 성공
- LLM-to-LLM 보완: {len(result.llm_to_llm_results)}번 시도, {sum(1 for r in result.llm_to_llm_results if r.success)}번 성공
- 보완으로 복구된 공격: {result.enhanced_attacks}개
"""

        if failure_analyses:
            most_common_failure = max(
                set(a.failure_reason.value for a in failure_analyses),
                key=lambda r: sum(1 for a in failure_analyses if a.failure_reason.value == r)
            )
            summary += f"\n주요 실패 원인: {most_common_failure}"
            
        return summary.strip() 