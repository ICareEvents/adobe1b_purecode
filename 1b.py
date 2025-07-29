import json
import os
import sys
import re
import time
import fitz
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import defaultdict

def sanitize_md_text(txt):
    txt = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', txt)
    txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
    txt = re.sub(r'\*(.*?)\*', r'\1', txt)
    txt = re.sub(r'_(.*?)_', r'\1', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

def locate_primary_font(font_word_tally):
    if not font_word_tally:
        return None
    return max(font_word_tally.items(), key=lambda x: x[1])[0]

def tally_font_words(font_data):
    word_count_map = defaultdict(int)
    
    for entry in font_data:
        word_list = entry["text"].split()
        word_count_map[entry["font_size"]] += len(word_list)
    
    return dict(word_count_map)

def gather_font_data(pg):
    font_elements = []
    page_blocks = pg.get_text("dict")
    
    for blk in page_blocks["blocks"]:
        if "lines" in blk:
            for ln in blk["lines"]:
                for segment in ln["spans"]:
                    if segment["text"].strip():
                        font_elements.append({
                            "text": segment["text"].strip(),
                            "font_size": round(segment["size"]),
                            "font": segment["font"],
                            "bbox": segment["bbox"]
                        })
    
    return font_elements

def segment_by_font(pg):
    font_data = gather_font_data(pg)
    if not font_data:
        return []
    
    word_tally = tally_font_words(font_data)
    primary_font = locate_primary_font(word_tally)
    
    if not primary_font:
        return []
    
    doc_sections = []
    active_section = {
        'title': '',
        'content': [],
        'font_size': 0
    }
    
    for entry in font_data:
        if entry["font_size"] > primary_font:
            if active_section['title'] and active_section['content']:
                doc_sections.append({
                    'title': active_section['title'],
                    'content': '\n'.join(active_section['content'])
                })
            
            active_section = {
                'title': entry["text"],
                'content': [],
                'font_size': entry["font_size"]
            }
        else:
            active_section['content'].append(entry["text"])
    
    if active_section['title'] and active_section['content']:
        doc_sections.append({
            'title': active_section['title'],
            'content': '\n'.join(active_section['content'])
        })
    elif active_section['content']:
        doc_sections.append({
            'title': 'Document Content',
            'content': '\n'.join(active_section['content'])
        })
    
    return doc_sections

def select_important_sentences(text_content, query_vec, model_instance, sent_count=3):
    sentence_list = re.split(r'(?<=[.!?])\s+', text_content.replace('\n', ' '))
    sentence_list = [s.strip() for s in sentence_list if len(s.strip()) > 15]
    
    if not sentence_list:
        return text_content[:200] + "..." if len(text_content) > 200 else text_content
    
    if len(sentence_list) <= sent_count:
        return " ".join(sentence_list)
    
    try:
        sent_vectors = model_instance.encode(sentence_list, convert_to_tensor=True)
        similarity_scores = util.cos_sim(query_vec, sent_vectors)
        
        scored_sents = list(zip(sentence_list, similarity_scores[0]))
        scored_sents.sort(key=lambda x: x[1], reverse=True)
        
        selected_sents = [s[0] for s in scored_sents[:sent_count]]
        
        ordered_sents = []
        for sent in sentence_list:
            if sent in selected_sents:
                ordered_sents.append(sent)
                if len(ordered_sents) == sent_count:
                    break
        
        return " ".join(ordered_sents)
    except Exception as e:
        print(f"Warning: Error in sentence extraction: {e}. Using first few sentences.", file=sys.stderr)
        return " ".join(sentence_list[:sent_count])

def create_synopsis(txt, summary_model, word_limit=100):
    try:
        processed_txt = re.sub(r'\s+', ' ', txt.strip())
        word_list = processed_txt.split()
        
        if len(word_list) <= 30:
            return processed_txt
        
        if len(word_list) > 1000:
            processed_txt = ' '.join(word_list[:1000])
        
        prompt_txt = f"summarize: {processed_txt}"
        
        input_word_cnt = len(processed_txt.split())
        adaptive_max_len = max(20, min(word_limit, int(input_word_cnt * 0.7)))
        
        synopsis_output = summary_model(prompt_txt, 
                                      max_length=adaptive_max_len,
                                      min_length=min(15, adaptive_max_len - 5),
                                      do_sample=False,
                                      truncation=True)
        
        synopsis = synopsis_output[0]['summary_text']
        
        synopsis_words = synopsis.split()
        if len(synopsis_words) > word_limit:
            synopsis = ' '.join(synopsis_words[:word_limit])
        
        return synopsis
        
    except Exception as e:
        print(f"Warning: Error generating summary: {e}. Using truncated original text.", file=sys.stderr)
        word_list = txt.split()
        return ' '.join(word_list[:word_limit]) + ("..." if len(word_list) > word_limit else "")

def parse_md_sections(md_content):
    doc_parts = []
    current_part = {
        'title': '',
        'content': []
    }
    
    for line_info in md_content:
        txt = line_info['text']
        
        if line_info['has_bold_underline']:
            if current_part['title'] and current_part['content']:
                doc_parts.append({
                    'title': current_part['title'],
                    'content': '\n'.join(current_part['content'])
                })
            
            current_part = {
                'title': txt,
                'content': []
            }
        
        elif line_info['has_bold'] and not current_part['title']:
            if current_part['content']:
                doc_parts.append({
                    'title': 'Introduction',
                    'content': '\n'.join(current_part['content'])
                })
            
            current_part = {
                'title': txt,
                'content': []
            }
        
        elif line_info['has_bold'] and len(txt.split()) <= 10 and not txt.endswith('.'):
            if current_part['title'] and current_part['content']:
                doc_parts.append({
                    'title': current_part['title'],
                    'content': '\n'.join(current_part['content'])
                })
            
            current_part = {
                'title': txt,
                'content': []
            }
        
        else:
            if txt and not (txt.startswith('**') and txt.endswith('**')):
                current_part['content'].append(txt)
    
    if current_part['title'] and current_part['content']:
        doc_parts.append({
            'title': current_part['title'],
            'content': '\n'.join(current_part['content'])
        })
    elif current_part['content']:
        doc_parts.append({
            'title': 'Document Content',
            'content': '\n'.join(current_part['content'])
        })
    
    return doc_parts

def transform_pdf_to_md(pg):
    page_blocks = pg.get_text("dict")["blocks"]
    md_content = []
    
    for blk in page_blocks:
        if "lines" in blk:
            for ln in blk["lines"]:
                line_txt = ""
                line_format = []
                
                for segment in ln["spans"]:
                    txt = segment["text"]
                    format_flags = segment["flags"]
                    
                    is_bold = bool(format_flags & 2**4)
                    is_italic = bool(format_flags & 2**3)
                    is_underline = bool(format_flags & 2**1)
                    
                    if txt.strip():
                        styled_txt = txt
                        if is_bold and is_underline:
                            styled_txt = f"***{txt.strip()}***"
                        elif is_bold:
                            styled_txt = f"**{txt.strip()}**"
                        elif is_italic:
                            styled_txt = f"*{txt.strip()}*"
                        elif is_underline:
                            styled_txt = f"_{txt.strip()}_"
                        
                        line_format.append({
                            'text': styled_txt,
                            'is_bold': is_bold,
                            'is_underline': is_underline,
                            'is_bold_underline': is_bold and is_underline
                        })
                
                if line_format:
                    merged_txt = " ".join([item['text'] for item in line_format])
                    if merged_txt.strip():
                        has_bold_under = any(item['is_bold_underline'] for item in line_format)
                        has_bold = any(item['is_bold'] for item in line_format)
                        
                        md_content.append({
                            'text': merged_txt.strip(),
                            'has_bold_underline': has_bold_under,
                            'has_bold': has_bold
                        })
    
    return md_content

def process_pdf_documents(pdf_dir, file_names):
    all_parts = []
    print("-> Parsing PDF documents using markdown conversion...", file=sys.stderr)
    
    for fname in file_names:
        complete_path = os.path.join(pdf_dir, fname)
        if not os.path.exists(complete_path):
            print(f"Warning: PDF file not found at {complete_path}. Skipping.", file=sys.stderr)
            continue

        try:
            pdf_doc = fitz.open(complete_path)
            for pg_num, pg in enumerate(pdf_doc):
                md_lines = transform_pdf_to_md(pg)
                
                if md_lines:
                    doc_sections = parse_md_sections(md_lines)
                    
                    if not doc_sections or len(doc_sections) == 1:
                        print(f"-> Markdown parsing yielded {len(doc_sections)} sections for page {pg_num + 1}, trying font-size based parsing...", file=sys.stderr)
                        doc_sections = segment_by_font(pg)
                    
                    for sect in doc_sections:
                        cleaned_title = sanitize_md_text(sect['title'])
                        cleaned_content = sanitize_md_text(sect['content'])
                        
                        if cleaned_content:
                            all_parts.append({
                                "document": fname,
                                "page_number": pg_num + 1,
                                "section_title": cleaned_title,
                                "content": cleaned_content,
                                "raw_markdown_title": sect['title'],
                                "raw_markdown_content": sect['content']
                            })
                else:
                    print(f"-> No markdown content found for page {pg_num + 1}, using font-size based parsing...", file=sys.stderr)
                    doc_sections = segment_by_font(pg)
                    
                    if doc_sections:
                        for sect in doc_sections:
                            if sect['content']:
                                all_parts.append({
                                    "document": fname,
                                    "page_number": pg_num + 1,
                                    "section_title": sect['title'],
                                    "content": sect['content'],
                                    "raw_markdown_title": sect['title'],
                                    "raw_markdown_content": sect['content']
                                })
                    else:
                        raw_txt = pg.get_text("text")
                        if raw_txt.strip():
                            opening_line = next((line for line in raw_txt.strip().split('\n') if line.strip()), "Document Content")
                            all_parts.append({
                                "document": fname,
                                "page_number": pg_num + 1,
                                "section_title": opening_line,
                                "content": raw_txt.strip(),
                                "raw_markdown_title": opening_line,
                                "raw_markdown_content": raw_txt.strip()
                            })
            pdf_doc.close()
        except Exception as e:
            print(f"Warning: Could not process {fname}. Error: {e}. Skipping.", file=sys.stderr)

    print(f"-> Finished parsing. Found {len(all_parts)} total sections.", file=sys.stderr)
    return all_parts

def execute_persona_task(input_json_file, pdf_dir):
    start_timestamp = time.time()
    print(f"-> Code execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    
    with open(input_json_file, 'r') as f:
        input_config = json.load(f)

    user_persona = input_config.get("persona", {}).get("role", "")
    task_objective = input_config.get("job_to_be_done", {}).get("task", "")
    doc_files = [doc.get("filename") for doc in input_config.get("documents", [])]
    
    search_query = f"{user_persona}: {task_objective}"
    print(f"-> Starting analysis for query: {search_query}\n", file=sys.stderr)

    model_storage = "model"
    os.makedirs(model_storage, exist_ok=True)
    
    print("-> Loading sentence-transformer model (paraphrase-MiniLM-L6-v2)...", file=sys.stderr)
    try:
        sentence_model_dir = os.path.join(model_storage, 'paraphrase-MiniLM-L6-v2')
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=model_storage)
        print("-> Sentence transformer model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}", file=sys.stderr)
        return

    print("-> Loading T5 summarization model (google/flan-t5-small)...", file=sys.stderr)
    try:
        summary_pipeline = pipeline("summarization", 
                            model="google/flan-t5-small",
                            device=-1,
                            model_kwargs={"cache_dir": model_storage})
        print("-> FLAN-T5 summarization model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading summarization model: {e}", file=sys.stderr)
        return

    document_sections = process_pdf_documents(pdf_dir, doc_files)
    if not document_sections:
        print("Error: No sections were extracted from PDFs. Cannot proceed.", file=sys.stderr)
        return

    print("-> Embedding query and document sections...", file=sys.stderr)
    try:
        query_vector = embedding_model.encode(search_query, convert_to_tensor=True)
        section_texts = [sect["content"] for sect in document_sections]
        section_vectors = embedding_model.encode(section_texts, convert_to_tensor=True)
        print("-> Embedding complete.", file=sys.stderr)
    except Exception as e:
        print(f"Error during embedding: {e}", file=sys.stderr)
        return

    print("-> Calculating relevance scores for section ranking...", file=sys.stderr)
    try:
        similarity_matrix = util.cos_sim(query_vector, section_vectors)

        for idx, sect in enumerate(document_sections):
            sect['relevance_score'] = similarity_matrix[0][idx].item()
        
        ordered_sections = sorted(document_sections, key=lambda x: x['relevance_score'], reverse=True)
    except Exception as e:
        print(f"Error calculating relevance scores: {e}", file=sys.stderr)
        return

    print("-> Performing sub-section analysis and summary generation on top sections...", file=sys.stderr)
    analysis_output = []
    priority_sections = ordered_sections[:5]

    for idx, sect in enumerate(priority_sections):
        print(f"-> Processing section {idx+1}/{len(priority_sections)}: {sect['section_title'][:50]}...", file=sys.stderr)
        try:
            section_synopsis = create_synopsis(sect['content'], summary_pipeline, word_limit=100)
            
            clean_title = sanitize_md_text(sect.get("raw_markdown_title", sect["section_title"]))
            
            analysis_output.append({
                "document": sect["document"],
                "refined_text": section_synopsis,
                "page_number": sect["page_number"]
            })
        except Exception as e:
            print(f"Warning: Error processing section '{sect['section_title']}': {e}", file=sys.stderr)
            backup_text = sect["content"][:750] + ("..." if len(sect["content"]) > 750 else "")
            clean_title = sanitize_md_text(sect.get("raw_markdown_title", sect["section_title"]))
            
            analysis_output.append({
                "document": sect["document"],  
                "refined_text": backup_text,
                "page_number": sect["page_number"]
            })

    print("-> Formatting final output JSON...", file=sys.stderr)
    result_data = {
        "metadata": {
            "input_documents": doc_files,
            "persona": user_persona,
            "job_to_be_done": task_objective,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": sect["document"],
                "section_title": sanitize_md_text(sect.get("raw_markdown_title", sect["section_title"])),
                "importance_rank": idx + 1,
                "page_number": sect["page_number"]
            } for idx, sect in enumerate(priority_sections)
        ],
        "subsection_analysis": analysis_output
    }

    output_file = "challenge1b_output.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        
        print(f"\nAnalysis complete. Output saved to {output_file}", file=sys.stderr)
        print(f"Found {len(document_sections)} sections total, analyzed top {len(priority_sections)} sections.", file=sys.stderr)
        print(f"Generated 100-word summaries for each analyzed section.", file=sys.stderr)
        
        end_timestamp = time.time()
        processing_duration = end_timestamp - start_timestamp
        print(f"-> Code execution ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
        print(f"-> Total execution time: {processing_duration:.2f} seconds", file=sys.stderr)
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

if __name__ == '__main__':
    config_file = "challenge1b_input.json"
    pdf_location = "Adobe-India-Hackathon25/Challenge_1b/Collection 1/PDFs"

    if not os.path.exists(config_file):
        print(f"Error: Input JSON file not found at {config_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(pdf_location):
        print(f"Error: PDF directory not found at {pdf_location}", file=sys.stderr)
        sys.exit(1)
    
    execute_persona_task(config_file, pdf_location)