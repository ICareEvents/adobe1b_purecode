import json
import os
import sys
from collections import defaultdict
import re
from datetime import datetime

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def normalize_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def calculate_keyword_relevance(text, keywords):
    if not text or not keywords:
        return 0.0
    
    normalized_text = normalize_text(text)
    normalized_keywords = [normalize_text(kw) for kw in keywords]
    
    text_words = set(normalized_text.split())
    keyword_matches = 0
    
    for keyword in normalized_keywords:
        if keyword in normalized_text:
            keyword_matches += 1
        else:
            keyword_words = set(keyword.split())
            if keyword_words.intersection(text_words):
                keyword_matches += 0.5
    
    return min(keyword_matches / len(keywords), 1.0) if keywords else 0.0

def evaluate_section_relevance(sections, persona, job_to_be_done, expected_keywords=None):
    if not sections:
        return 0.0
    
    if expected_keywords is None:
        expected_keywords = extract_keywords_from_persona_job(persona, job_to_be_done)
    
    relevance_scores = []
    
    for section in sections:
        section_title = section.get('section_title', '')
        importance_rank = section.get('importance_rank', float('inf'))
        
        title_relevance = calculate_keyword_relevance(section_title, expected_keywords)
        
        rank_score = max(0, (6 - importance_rank) / 5) if importance_rank <= 5 else 0
        
        combined_score = (title_relevance * 0.7) + (rank_score * 0.3)
        relevance_scores.append(combined_score)
    
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

def evaluate_subsection_quality(subsections, persona, job_to_be_done):
    if not subsections:
        return 0.0
    
    quality_scores = []
    
    for subsection in subsections:
        refined_text = subsection.get('refined_text', '')
        
        if len(refined_text.strip()) < 20:
            quality_scores.append(0.0)
            continue
        
        text_length_score = min(len(refined_text.split()) / 100, 1.0)
        
        persona_keywords = extract_keywords_from_text(persona)
        job_keywords = extract_keywords_from_text(job_to_be_done)
        
        persona_relevance = calculate_keyword_relevance(refined_text, persona_keywords)
        job_relevance = calculate_keyword_relevance(refined_text, job_keywords)
        
        coherence_score = calculate_text_coherence(refined_text)
        
        combined_score = (
            text_length_score * 0.2 +
            persona_relevance * 0.3 +
            job_relevance * 0.3 +
            coherence_score * 0.2
        )
        
        quality_scores.append(combined_score)
    
    return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

def extract_keywords_from_persona_job(persona, job_to_be_done):
    combined_text = f"{persona} {job_to_be_done}"
    return extract_keywords_from_text(combined_text)

def extract_keywords_from_text(text):
    if not text:
        return []
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words]
    
    word_freq = defaultdict(int)
    for word in keywords:
        word_freq[word] += 1
    
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:10]]

def calculate_text_coherence(text):
    if not text or len(text.strip()) < 10:
        return 0.0
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.5
    
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    length_score = min(avg_sentence_length / 15, 1.0) if avg_sentence_length > 0 else 0.0
    
    sentence_variety = len(set(len(s.split()) for s in sentences)) / len(sentences)
    variety_score = min(sentence_variety * 2, 1.0)
    
    return (length_score * 0.6) + (variety_score * 0.4)

def validate_output_format(output_data):
    required_metadata_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
    required_section_fields = ['document', 'section_title', 'importance_rank', 'page_number']
    required_subsection_fields = ['document', 'refined_text', 'page_number']
    
    issues = []
    
    if 'metadata' not in output_data:
        issues.append("Missing 'metadata' field")
    else:
        metadata = output_data['metadata']
        for field in required_metadata_fields:
            if field not in metadata:
                issues.append(f"Missing metadata field: {field}")
    
    if 'extracted_sections' not in output_data:
        issues.append("Missing 'extracted_sections' field")
    else:
        sections = output_data['extracted_sections']
        if not isinstance(sections, list):
            issues.append("'extracted_sections' should be a list")
        else:
            for i, section in enumerate(sections):
                for field in required_section_fields:
                    if field not in section:
                        issues.append(f"Missing field '{field}' in section {i}")
    
    if 'subsection_analysis' not in output_data:
        issues.append("Missing 'subsection_analysis' field")
    else:
        subsections = output_data['subsection_analysis']
        if not isinstance(subsections, list):
            issues.append("'subsection_analysis' should be a list")
        else:
            for i, subsection in enumerate(subsections):
                for field in required_subsection_fields:
                    if field not in subsection:
                        issues.append(f"Missing field '{field}' in subsection {i}")
    
    return issues

def evaluate_single_output(output_path, test_case_info=None):
    output_data = load_json_file(output_path)
    if not output_data:
        return None
    
    results = {
        'format_validation': [],
        'section_relevance_score': 0.0,
        'subsection_quality_score': 0.0,
        'total_sections': 0,
        'total_subsections': 0,
        'ranking_consistency': 0.0,
        'overall_score': 0.0
    }
    
    format_issues = validate_output_format(output_data)
    results['format_validation'] = format_issues
    
    if format_issues:
        print(f"Format validation issues found: {len(format_issues)}")
        for issue in format_issues:
            print(f"  - {issue}")
        return results
    
    metadata = output_data.get('metadata', {})
    persona = metadata.get('persona', '')
    job_to_be_done = metadata.get('job_to_be_done', '')
    
    sections = output_data.get('extracted_sections', [])
    subsections = output_data.get('subsection_analysis', [])
    
    results['total_sections'] = len(sections)
    results['total_subsections'] = len(subsections)
    
    expected_keywords = None
    if test_case_info:
        expected_keywords = test_case_info.get('expected_keywords')
    
    results['section_relevance_score'] = evaluate_section_relevance(
        sections, persona, job_to_be_done, expected_keywords
    )
    
    results['subsection_quality_score'] = evaluate_subsection_quality(
        subsections, persona, job_to_be_done
    )
    
    ranking_scores = []
    for section in sections:
        rank = section.get('importance_rank', 0)
        if 1 <= rank <= 5:
            ranking_scores.append((6 - rank) / 5)
    
    results['ranking_consistency'] = sum(ranking_scores) / len(ranking_scores) if ranking_scores else 0.0
    
    section_score = results['section_relevance_score'] * 60
    subsection_score = results['subsection_quality_score'] * 40
    results['overall_score'] = section_score + subsection_score
    
    return results

def run_comprehensive_test_1b(output_dir, test_cases_file=None):
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return
    
    test_cases = {}
    if test_cases_file and os.path.exists(test_cases_file):
        test_cases = load_json_file(test_cases_file) or {}
    
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    overall_results = {
        'total_files': 0,
        'successful_evaluations': 0,
        'average_section_relevance': 0.0,
        'average_subsection_quality': 0.0,
        'average_overall_score': 0.0,
        'file_results': {}
    }
    
    print("Challenge 1B - Persona-Driven Document Intelligence Evaluation")
    print("=" * 60)
    
    for output_file in output_files:
        output_path = os.path.join(output_dir, output_file)
        test_case_info = test_cases.get(output_file.replace('.json', ''))
        
        results = evaluate_single_output(output_path, test_case_info)
        
        if results and not results['format_validation']:
            overall_results['total_files'] += 1
            overall_results['successful_evaluations'] += 1
            overall_results['file_results'][output_file] = results
            
            overall_results['average_section_relevance'] += results['section_relevance_score']
            overall_results['average_subsection_quality'] += results['subsection_quality_score']
            overall_results['average_overall_score'] += results['overall_score']
            
            print(f"\nFile: {output_file}")
            print(f"Section Relevance Score: {results['section_relevance_score']:.3f}")
            print(f"Subsection Quality Score: {results['subsection_quality_score']:.3f}")
            print(f"Overall Score: {results['overall_score']:.1f}/100")
            print(f"Total Sections: {results['total_sections']}")
            print(f"Total Subsections: {results['total_subsections']}")
            print(f"Ranking Consistency: {results['ranking_consistency']:.3f}")
        else:
            overall_results['total_files'] += 1
            if results and results['format_validation']:
                print(f"\nFile: {output_file} - Format validation failed")
            else:
                print(f"\nFile: {output_file} - Failed to evaluate")
    
    if overall_results['successful_evaluations'] > 0:
        n = overall_results['successful_evaluations']
        overall_results['average_section_relevance'] /= n
        overall_results['average_subsection_quality'] /= n
        overall_results['average_overall_score'] /= n
        
        print("\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        print(f"Total Files Processed: {overall_results['total_files']}")
        print(f"Successful Evaluations: {overall_results['successful_evaluations']}")
        print(f"Average Section Relevance: {overall_results['average_section_relevance']:.3f}")
        print(f"Average Subsection Quality: {overall_results['average_subsection_quality']:.3f}")
        print(f"Average Overall Score: {overall_results['average_overall_score']:.1f}/100")
        
        section_points = overall_results['average_section_relevance'] * 60
        subsection_points = overall_results['average_subsection_quality'] * 40
        
        print(f"\nScore Breakdown:")
        print(f"  Section Relevance (60 pts): {section_points:.1f}")
        print(f"  Sub-Section Quality (40 pts): {subsection_points:.1f}")
        print(f"  Total Estimated Score: {section_points + subsection_points:.1f}/100")
    
    return overall_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_1b.py <output_dir> [test_cases_file]")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    test_cases_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_comprehensive_test_1b(output_dir, test_cases_file)