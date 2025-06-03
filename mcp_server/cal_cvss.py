#!/usr/bin/env python3
import sys
import json
from typing import Dict, Any, Union
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(
    "CVSS_Calculator_MCP",
    instructions="A CVSS (Common Vulnerability Scoring System) calculator that computes vulnerability scores based on standard metrics.",
    host="0.0.0.0",
    port=1110,
)

def log_error(message: str):
    """Log error messages to stderr for debugging"""
    print(f"ERROR: {message}", file=sys.stderr)

def log_info(message: str):
    """Log informational messages to stderr for debugging"""
    print(f"INFO: {message}", file=sys.stderr)

def calculate_cvss(metrics: Dict[str, str]) -> float:
    """
    Calculate the CVSS base score based on the provided metrics.

    :param metrics: A dictionary containing the CVSS metrics with keys:
                    'AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A'
    :return: The CVSS base score as a float.
    """
    exploitability_coefficient = 8.22
    scope_coefficient = 1.08

    # Define associative arrays mapping each metric value to the constant used in the CVSS scoring formula.
    weight = {
        'AV': {'N': 0.85, 'A': 0.62, 'L': 0.55, 'P': 0.2},
        'AC': {'H': 0.44, 'L': 0.77},
        'PR': {
            'U': {'N': 0.85, 'L': 0.62, 'H': 0.27},  # Scope Unchanged
            'C': {'N': 0.85, 'L': 0.68, 'H': 0.5}    # Scope Changed
        },
        'UI': {'N': 0.85, 'R': 0.62},
        'S': {'U': 6.42, 'C': 7.52},
        'C': {'N': 0, 'L': 0.22, 'H': 0.56},
        'I': {'N': 0, 'L': 0.22, 'H': 0.56},
        'A': {'N': 0, 'L': 0.22, 'H': 0.56}
    }

    def round_up(input_value):
        """Round up to the nearest 0.1 as per the CVSS specification."""
        int_input = int(input_value * 100000)
        if int_input % 10000 == 0:
            return int_input / 100000
        else:
            return (int(int_input / 10000) + 1) / 10

    try:
        # Extract metric weights
        metric_weight = {key: weight[key][metrics[key]] for key in ['AV', 'AC', 'UI', 'C', 'I', 'A']}
        metric_weight['PR'] = weight['PR'][metrics['S']][metrics['PR']]
        metric_weight['S'] = weight['S'][metrics['S']]

        # Calculate impact sub-score
        impact_sub_score_multiplier = (1 - ((1 - metric_weight['C']) * (1 - metric_weight['I']) * (1 - metric_weight['A'])))
        if metrics['S'] == 'U':
            impact_sub_score = metric_weight['S'] * impact_sub_score_multiplier
        else:
            impact_sub_score = metric_weight['S'] * (impact_sub_score_multiplier - 0.029) - \
                               3.25 * (impact_sub_score_multiplier - 0.02) ** 15

        # Calculate exploitability sub-score
        exploitability_sub_score = exploitability_coefficient * metric_weight['AV'] * metric_weight['AC'] * \
                                   metric_weight['PR'] * metric_weight['UI']

        # Calculate base score
        if impact_sub_score <= 0:
            base_score = 0
        else:
            if metrics['S'] == 'U':
                base_score = min((exploitability_sub_score + impact_sub_score), 10)
            else:
                base_score = min((exploitability_sub_score + impact_sub_score) * scope_coefficient, 10)

        # Round up to one decimal place
        return round_up(base_score)

    except KeyError as e:
        raise ValueError(f"Invalid metric value: {e}")

def validate_cvss_metrics(metrics: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate CVSS metrics and return validation results.
    
    :param metrics: Dictionary containing CVSS metrics
    :return: Dictionary with validation results
    """
    required_metrics = ['AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A']
    valid_values = {
        'AV': ['N', 'A', 'L', 'P'],  # Network, Adjacent, Local, Physical
        'AC': ['H', 'L'],             # High, Low
        'PR': ['N', 'L', 'H'],        # None, Low, High
        'UI': ['N', 'R'],             # None, Required
        'S': ['U', 'C'],              # Unchanged, Changed
        'C': ['N', 'L', 'H'],         # None, Low, High
        'I': ['N', 'L', 'H'],         # None, Low, High
        'A': ['N', 'L', 'H']          # None, Low, High
    }
    
    validation_result = {
        'is_valid': True,
        'missing_metrics': [],
        'invalid_values': [],
        'errors': []
    }
    
    # Check for missing metrics
    for metric in required_metrics:
        if metric not in metrics:
            validation_result['missing_metrics'].append(metric)
            validation_result['is_valid'] = False
    
    # Check for invalid values
    for metric, value in metrics.items():
        if metric in valid_values:
            if value not in valid_values[metric]:
                validation_result['invalid_values'].append({
                    'metric': metric,
                    'value': value,
                    'valid_values': valid_values[metric]
                })
                validation_result['is_valid'] = False
        else:
            validation_result['errors'].append(f"Unknown metric: {metric}")
            validation_result['is_valid'] = False
    
    return validation_result

def get_severity_rating(score: float) -> str:
    """
    Get severity rating based on CVSS score.
    
    :param score: CVSS base score
    :return: Severity rating string
    """
    if score == 0.0:
        return "None"
    elif 0.1 <= score <= 3.9:
        return "Low"
    elif 4.0 <= score <= 6.9:
        return "Medium"
    elif 7.0 <= score <= 8.9:
        return "High"
    elif 9.0 <= score <= 10.0:
        return "Critical"
    else:
        return "Invalid"

@mcp.tool()
async def calculate_cvss_score(
    AV: str,  # Attack Vector: N(Network), A(Adjacent), L(Local), P(Physical)
    AC: str,  # Attack Complexity: H(High), L(Low)
    PR: str,  # Privileges Required: N(None), L(Low), H(High)
    UI: str,  # User Interaction: N(None), R(Required)
    S: str,   # Scope: U(Unchanged), C(Changed)
    C: str,   # Confidentiality: N(None), L(Low), H(High)
    I: str,   # Integrity: N(None), L(Low), H(High)
    A: str    # Availability: N(None), L(Low), H(High)
) -> str:
    """CVSS v3.1 기반으로 취약점의 기본 점수를 계산합니다.
    
    Args:
        AV: 공격 벡터 (N=네트워크, A=인접, L=로컬, P=물리적)
        AC: 공격 복잡성 (H=높음, L=낮음)
        PR: 필요 권한 (N=없음, L=낮음, H=높음)
        UI: 사용자 상호작용 (N=없음, R=필요)
        S: 범위 (U=변경없음, C=변경됨)
        C: 기밀성 영향 (N=없음, L=낮음, H=높음)
        I: 무결성 영향 (N=없음, L=낮음, H=높음)
        A: 가용성 영향 (N=없음, L=낮음, H=높음)
        
    Returns:
        CVSS 점수와 심각도 등급이 포함된 결과
    """
    try:
        metrics = {
            'AV': AV.upper(),
            'AC': AC.upper(),
            'PR': PR.upper(),
            'UI': UI.upper(),
            'S': S.upper(),
            'C': C.upper(),
            'I': I.upper(),
            'A': A.upper()
        }
        
        # Validate metrics first
        validation = validate_cvss_metrics(metrics)
        if not validation['is_valid']:
            error_details = []
            if validation['missing_metrics']:
                error_details.append(f"Missing metrics: {', '.join(validation['missing_metrics'])}")
            if validation['invalid_values']:
                for invalid in validation['invalid_values']:
                    error_details.append(f"Invalid value '{invalid['value']}' for {invalid['metric']}. Valid values: {', '.join(invalid['valid_values'])}")
            if validation['errors']:
                error_details.extend(validation['errors'])
            
            return f"Validation failed:\n" + "\n".join(error_details)
        
        # Calculate score
        score = calculate_cvss(metrics)
        severity = get_severity_rating(score)
        
        # Create detailed result
        result = {
            'cvss_score': score,
            'severity_rating': severity,
            'metrics_used': metrics,
            'score_interpretation': {
                'None': '0.0',
                'Low': '0.1 - 3.9',
                'Medium': '4.0 - 6.9', 
                'High': '7.0 - 8.9',
                'Critical': '9.0 - 10.0'
            }
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        log_error(f"Error calculating CVSS: {str(e)}")
        return f"Error calculating CVSS score: {str(e)}"

@mcp.tool()
async def validate_cvss_input(
    AV: str = None,
    AC: str = None,
    PR: str = None,
    UI: str = None,
    S: str = None,
    C: str = None,
    I: str = None,
    A: str = None
) -> str:
    """CVSS 메트릭 입력값의 유효성을 검증합니다.
    
    Args:
        AV: 공격 벡터 (선택사항)
        AC: 공격 복잡성 (선택사항)
        PR: 필요 권한 (선택사항)
        UI: 사용자 상호작용 (선택사항)
        S: 범위 (선택사항)
        C: 기밀성 영향 (선택사항)
        I: 무결성 영향 (선택사항)
        A: 가용성 영향 (선택사항)
        
    Returns:
        유효성 검증 결과와 가능한 값들의 목록
    """
    try:
        # Collect provided metrics
        provided_metrics = {}
        for metric_name, metric_value in [('AV', AV), ('AC', AC), ('PR', PR), ('UI', UI), ('S', S), ('C', C), ('I', I), ('A', A)]:
            if metric_value is not None:
                provided_metrics[metric_name] = metric_value.upper()
        
        # Validate provided metrics
        validation = validate_cvss_metrics(provided_metrics)
        
        # Prepare result
        result = {
            'validation_result': validation,
            'metric_definitions': {
                'AV (Attack Vector)': {
                    'description': '공격 벡터 - 취약점이 악용되는 경로',
                    'values': {
                        'N': 'Network - 네트워크를 통한 원격 접근',
                        'A': 'Adjacent - 인접 네트워크를 통한 접근',
                        'L': 'Local - 로컬 접근 (물리적 또는 논리적)',
                        'P': 'Physical - 물리적 접근 필요'
                    }
                },
                'AC (Attack Complexity)': {
                    'description': '공격 복잡성 - 공격 성공에 필요한 조건의 복잡성',
                    'values': {
                        'L': 'Low - 낮은 복잡성, 반복 가능한 공격',
                        'H': 'High - 높은 복잡성, 특별한 조건 필요'
                    }
                },
                'PR (Privileges Required)': {
                    'description': '필요 권한 - 공격자가 사전에 가져야 할 권한 수준',
                    'values': {
                        'N': 'None - 권한 불필요',
                        'L': 'Low - 기본 사용자 권한',
                        'H': 'High - 관리자 권한'
                    }
                },
                'UI (User Interaction)': {
                    'description': '사용자 상호작용 - 공격 성공을 위한 사용자 개입 필요성',
                    'values': {
                        'N': 'None - 사용자 개입 불필요',
                        'R': 'Required - 사용자 개입 필요'
                    }
                },
                'S (Scope)': {
                    'description': '범위 - 취약점의 영향이 다른 구성요소로 확산되는지 여부',
                    'values': {
                        'U': 'Unchanged - 영향 범위 변경 없음',
                        'C': 'Changed - 다른 구성요소로 영향 확산'
                    }
                },
                'C (Confidentiality)': {
                    'description': '기밀성 영향 - 정보 누출 정도',
                    'values': {
                        'N': 'None - 기밀성 영향 없음',
                        'L': 'Low - 부분적 정보 누출',
                        'H': 'High - 전체 정보 누출'
                    }
                },
                'I (Integrity)': {
                    'description': '무결성 영향 - 데이터 변조 가능성',
                    'values': {
                        'N': 'None - 무결성 영향 없음',
                        'L': 'Low - 부분적 데이터 변조',
                        'H': 'High - 전체 데이터 변조'
                    }
                },
                'A (Availability)': {
                    'description': '가용성 영향 - 서비스 중단 정도',
                    'values': {
                        'N': 'None - 가용성 영향 없음',
                        'L': 'Low - 부분적 서비스 영향',
                        'H': 'High - 전체 서비스 중단'
                    }
                }
            }
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        log_error(f"Error validating CVSS input: {str(e)}")
        return f"Error validating CVSS input: {str(e)}"

@mcp.tool()
async def get_cvss_info() -> str:
    """CVSS (Common Vulnerability Scoring System)에 대한 일반 정보를 제공합니다.
    
    Returns:
        CVSS 시스템에 대한 상세 정보
    """
    try:
        info = {
            'cvss_version': 'CVSS v3.1',
            'description': 'CVSS는 보안 취약점의 주요 특성을 포착하고 심각도를 숫자로 나타내는 공개 프레임워크입니다.',
            'score_ranges': {
                'None': '0.0',
                'Low': '0.1 - 3.9',
                'Medium': '4.0 - 6.9',
                'High': '7.0 - 8.9',
                'Critical': '9.0 - 10.0'
            },
            'metric_groups': {
                'Base Metrics': {
                    'description': '취약점의 고유한 특성으로 시간이 지나도 변하지 않는 속성',
                    'metrics': ['AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A']
                },
                'Temporal Metrics': {
                    'description': '시간에 따라 변할 수 있는 취약점의 특성 (현재 미구현)',
                    'metrics': ['E', 'RL', 'RC']
                },
                'Environmental Metrics': {
                    'description': '특정 환경에서의 취약점 영향 (현재 미구현)',
                    'metrics': ['CR', 'IR', 'AR', 'MAV', 'MAC', 'MPR', 'MUI', 'MS', 'MC', 'MI', 'MA']
                }
            },
            'calculation_formula': {
                'base_score': 'Exploitability × Impact (with scope adjustments)',
                'exploitability': '8.22 × AttackVector × AttackComplexity × PrivilegesRequired × UserInteraction',
                'impact': 'Scope × [1 - (1-Confidentiality) × (1-Integrity) × (1-Availability)]'
            },
            'usage_examples': [
                {
                    'scenario': '원격 코드 실행 취약점',
                    'metrics': 'AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H',
                    'expected_score': '10.0 (Critical)'
                },
                {
                    'scenario': '정보 누출 취약점',
                    'metrics': 'AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N',
                    'expected_score': '6.5 (Medium)'
                },
                {
                    'scenario': '로컬 권한 상승',
                    'metrics': 'AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H',
                    'expected_score': '7.8 (High)'
                }
            ],
            'references': [
                'CVSS v3.1 Specification: https://www.first.org/cvss/v3.1/specification-document',
                'CVSS Calculator: https://www.first.org/cvss/calculator/3.1',
                'NIST NVD: https://nvd.nist.gov/vuln-metrics/cvss'
            ]
        }
        
        return json.dumps(info, indent=2, ensure_ascii=False)
        
    except Exception as e:
        log_error(f"Error getting CVSS info: {str(e)}")
        return f"Error getting CVSS information: {str(e)}"

@mcp.tool()
async def calculate_multiple_scenarios(scenarios: str) -> str:
    """여러 CVSS 시나리오를 일괄 계산합니다.
    
    Args:
        scenarios: JSON 형태의 시나리오 목록. 각 시나리오는 name과 metrics를 포함해야 합니다.
                  예: [{"name": "Scenario 1", "metrics": {"AV": "N", "AC": "L", ...}}, ...]
        
    Returns:
        각 시나리오별 CVSS 점수 계산 결과
    """
    try:
        # Parse scenarios
        scenarios_data = json.loads(scenarios)
        
        if not isinstance(scenarios_data, list):
            return "Error: scenarios must be a list of scenario objects"
        
        results = []
        
        for i, scenario in enumerate(scenarios_data):
            try:
                scenario_name = scenario.get('name', f'Scenario {i+1}')
                metrics = scenario.get('metrics', {})
                
                # Validate required metrics
                required_metrics = ['AV', 'AC', 'PR', 'UI', 'S', 'C', 'I', 'A']
                missing_metrics = [m for m in required_metrics if m not in metrics]
                
                if missing_metrics:
                    results.append({
                        'name': scenario_name,
                        'error': f"Missing metrics: {', '.join(missing_metrics)}"
                    })
                    continue
                
                # Convert to uppercase
                normalized_metrics = {k: v.upper() for k, v in metrics.items()}
                
                # Validate metrics
                validation = validate_cvss_metrics(normalized_metrics)
                if not validation['is_valid']:
                    error_details = []
                    if validation['invalid_values']:
                        for invalid in validation['invalid_values']:
                            error_details.append(f"Invalid {invalid['metric']}: {invalid['value']}")
                    results.append({
                        'name': scenario_name,
                        'error': '; '.join(error_details)
                    })
                    continue
                
                # Calculate score
                score = calculate_cvss(normalized_metrics)
                severity = get_severity_rating(score)
                
                results.append({
                    'name': scenario_name,
                    'cvss_score': score,
                    'severity_rating': severity,
                    'metrics': normalized_metrics
                })
                
            except Exception as e:
                results.append({
                    'name': scenario.get('name', f'Scenario {i+1}'),
                    'error': str(e)
                })
        
        return json.dumps({
            'total_scenarios': len(scenarios_data),
            'results': results
        }, indent=2, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        return f"Error parsing scenarios JSON: {str(e)}"
    except Exception as e:
        log_error(f"Error calculating multiple scenarios: {str(e)}")
        return f"Error calculating multiple scenarios: {str(e)}"

def main():
    """pip을 통해 패키지가 설치될 때의 진입점"""
    log_info("Starting CVSS Calculator MCP Server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
