import re
import json
from bs4 import BeautifulSoup, Tag, NavigableString

abbr_map = {
    "ACT":   "Acceptance and commitment therapy",
    "ADHD":  "Attention-deficit hyperactivity disorder",
    "AI":    "Artificial intelligence",
    "BA":    "Behavioural activation",
    "CAM":   "Complementary and alternative medicine",
    "CANMAT":"Canadian Network for Mood and Anxiety Treatments",
    "CBASP": "Cognitive behavioural analysis system of psychotherapy",
    "CBT":   "Cognitive-behavioural therapy",
    "CPD":   "Continuing professional development",
    "CYP":   "Cytochrome P450",
    "DBS":   "Deep brain stimulation",
    "DHI":   "Digital health intervention",
    "DLPFC": "Dorsolateral prefrontal cortex",
    "DSM-5-TR": "Diagnostic and Statistical Manual, 5th edition, Text Revision",
    "DSM-IV-TR":"Diagnostic and Statistical Manual, 4th edition, Text Revision",
    "DTD":   "Difficult-to-treat depression",
    "ECG":   "Electrocardiography",
    "ECT":   "Electroconvulsive therapy",
    "EEG":   "Electroencephalography",
    "GRADE": "Grading of Recommendations Assessment, Development, and Evaluation",
    "ICD":   "International Classification of Diseases",
    "IPT":   "Interpersonal therapy",
    "MAOI":  "Monoamine oxidase inhibitor",
    "MBC":   "Measurement-based care",
    "MBCT":  "Mindfulness-based cognitive therapy",
    "MCT":   "Metacognitive therapy",
    "MDD":   "Major depressive disorder",
    "MDE":   "Major depressive episode",
    "MI":    "Motivational interviewing",
    "MST":   "Magnetic seizure therapy",
    "NbN":   "Neuroscience-based nomenclature",
    "NDRI":  "Norepinephrine-dopamine reuptake inhibitor",
    "NMDA":  "N-methyl-D-aspartate",
    "NSAID": "Nonsteroidal anti-inflammatory drug",
    "PDD":   "Persistent depressive disorder",
    "PDT":   "Psychodynamic psychotherapy",
    "PHQ":   "Patient health questionnaire",
    "PST":   "Problem-solving therapy",
    "RCT":   "Randomized controlled trial",
    "rTMS":  "Repetitive transcranial magnetic stimulation",
    "SDM":   "Shared decision-making",
    "SNRI":  "Serotonin-norepinephrine reuptake inhibitor",
    "SSRI":  "Selective serotonin reuptake inhibitor",
    "STPP":  "Short-term psychodynamic psychotherapy",
    "TBS":   "Theta burst stimulation",
    "TCA":   "Tricyclic antidepressants",
    "tDCS":  "Transcranial direct current stimulation",
    "TMS":   "Transcranial magnetic stimulation",
    "TRD":   "Treatment-resistant depression",
    "USA":   "United States of America",
    "VNS":   "Vagus nerve stimulation",
    "WHO":   "World Health Organization",
    }

# append definition to abbreviation

def append_definition(guideline):
    pattern = re.compile(r'\b([A-Z]{2,})\b')

    def replacer(match):
        abbr = match.group(1)
        if abbr in abbr_map:
            return f"{abbr} ({abbr_map[abbr]})"
        return abbr

    for i in range(len(guideline)):
        if guideline[i]['metadata']['referee_id'] == 'table_c':
            continue
        text = guideline[i]['text']
        text = pattern.sub(replacer, text)
        guideline[i]['text'] = text

    return guideline

def find_graph_by_id(graphs, graph_id):
    """
    Find a graph by its ID in the graphs dictionary.
    """
    for graph in graphs:
        if graph['metadata']['referee_id'] == graph_id:
            return graph['text']
        
    return None
    
    

def parse_title(soup):
    """
    Parse the title 
    """
    title = soup.find("h1")
    if title:
        title = title.decode_contents().replace('\n', '')
    chunk = {
    "text": title,
    "metadata":
        {
        "section": "title",
        "type": "title",
        "headings": "Title of the guideline document",
        "referenced_tables": [],
        "referee_id": ""
        }
    }
    return chunk

def merge_boxes(guideline):
    # if the item in the guideline is a box, merge it with the previous box. and remove the current box.
    # If there's no previous box, continue to the next item.
    previous_box = None
    for item in guideline[:]:
        if item['metadata']['type'] == 'box':
            if previous_box is None:
                previous_box = item
            else:
                    # if current chunk's referee_id is the same as previous box's referee_id, merge them
                    if item['metadata']['referee_id'] == previous_box['metadata']['referee_id']:
                        # merge the current box with the previous one
                        previous_box['text'] += "\n" + item['text']
                        previous_box['metadata']['referenced_tables'].extend(item['metadata']['referenced_tables'])
                        #remove the current item from the guideline
                        guideline.remove(item)
        else:
            if previous_box is not None:
                # if the previous box is not None, reset it to None
                previous_box = None

    

def prepend_headings_to_text(guideline):
    """give chunk_id to each chunk in the guideline and prepend headings to text"""
    for i in range(len(guideline)):
        guideline[i]['metadata']['chunk_id'] = i
        guideline[i]['text'] = guideline[i]['metadata']['headings'] + " > paragraph id: " + str(i) + "\n\n" + guideline[i]['text']
        
    # return guideline


def parse_main_article(soup, graphs, output):
    ## I manually substitute the Level 1, Level 2, Level 3, Level 4 link with text like (Leve 1), (Level 2), (Level 3), (Level 4)
    level1 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/62befe587468/10.1177_07067437241245384-img1.jpg'
    level2 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/b9ea5ad77490/10.1177_07067437241245384-img2.jpg'
    level3 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/5be38aafe33f/10.1177_07067437241245384-img3.jpg'
    level4 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/68e56cd87632/10.1177_07067437241245384-img4.jpg'
    
    # parse the main body
    for p in soup.find_all("p"):
        referenced_tables = set()
        referee_id = ""
        sec_id = ""
        
        #-----------------------replace the <img> tags---------------------
        # we also manually delete the duplication in first occurrence mentioning Levels
        for img in p.find_all('img'):
            src = img.get('src')
            if src == level1:
                replacement_text = "(Level 1)"
            elif src == level2:
                replacement_text = "(Level 2)"
            elif src == level3:
                replacement_text = "(Level 3)"
            elif src == level4:
                replacement_text = "(Level 4)"
            else:
                continue    
                
            text_node = NavigableString(replacement_text)
            img.replace_with(text_node)
            referenced_tables.add('table_a')


            
            
        # ----------------------get section id----------------------------------
        parent_sec = p.find_parent(["section",'figure'], id=True)
        sec_id = parent_sec.get("id") if parent_sec else None
        
        
        #-----------------------get the headings---------------------
        # special case: # manually finding the "No heading" in the html file to fix the No heading issue
        # delete the <div><\div> outside this to get the correct heading: <p>Protracted Discontinuation Symptoms and Hyperbolic Tapering Schedules.</p>
        #it's mostly the "Box"
        
        # get the closest heading
        heading = p.find_previous_sibling(lambda tag: bool(re.match(r'^h[2-6]$', tag.name)))
        headings = heading.get_text(strip=True) if heading else 'No heading' 

        
        if 'fig' in sec_id or 'table' in sec_id or 'box' in sec_id:
            referee_id = headings.lower().replace(".", " ").strip().replace(" ", "_")
        #while parent still has parents
        while parent_sec:
            # print(f"parent_sec: {parent_sec.get('id')}")
            heading = parent_sec.find_previous_sibling(lambda tag: bool(re.match(r'^h[2-6]$', tag.name)))
            if heading:
                headings = heading.get_text(strip=True) + ' > ' + headings
            parent_sec = parent_sec.find_parent("section", id=True)

        headings = headings.strip().replace('\n', ' ')
        

        #-----------------------get the text---------------------
        text = p.get_text(separator=' ', strip=True) # get only text
        
        
        #----------------------- mark Type of the text ------------------------
        type = 'paragraph'
        
        if 'table' in sec_id:  # either table image or table in HTML format
            if p.get('class') and 'img-box' in p.get('class'):
                type = 'table image '
            else: 
                continue
            ##################################
            try:
                text = find_graph_by_id(graphs, referee_id)
            except Exception as e:
                print(f"Error finding graph by ID: {e}")
                continue
            ##################################

            # I only have 2 here with parent = <figure>, most of img's parent are <section id = 'table...'>
        elif p.get('class') and 'img-box' in p.get('class'):
            type = 'figure image'
            ##################################
            try:
                text = find_graph_by_id(graphs, referee_id)
            except Exception as e:
                print(f"Error finding graph by ID: {e}")
                continue
            ##################################
        elif 'box' in sec_id: 
            type = 'box'
        
        if 'fig' in sec_id and type == 'paragraph':
            continue  # skip paragraphs in figures cause they are already handled in the figure processing
        
        #----------------------- get referenced tables ------------------------
        if not referee_id:
            all_links = p.find_all('a')
            for link in all_links:
                href = link.get('href')
                if href.startswith('#'):
                    table_id = link.get_text(strip=True).lower().replace(".", " ").strip().replace(" ", "_")
                    referenced_tables.add(table_id)
        
                
        #----------------------- formate the chunks ------------------------
        chunk = {
            "text": text,
            "metadata": {
            "section": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11351064/#" + sec_id,
            "type": type,
            "headings": headings,
            "referenced_tables": list(referenced_tables),
            "referee_id": referee_id,
            }
        }
        
        output.append(chunk)
    return output

    
def main():
    with open("data/processed/graphs.json", "r", encoding="utf-8") as f:
        graphs = json.load(f)
    with open("data/processed/tables.json", "r", encoding="utf-8") as f2:
        tables = json.load(f2)
        
    with open('data/processed/tables.json', 'r', encoding='utf-8') as f2:
        tables = json.load(f2)

# ----------------------- parse the html file ------------------------
    filename = "data/raw/source.html"
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    output = []
    
    output.append(parse_title(soup))
    output = parse_main_article(soup, graphs, output)
    print(f"Parsed {len(output)} chunks from the main article.")
    # output = merge_boxes(output)
    

    # ----------------------- write to json ------------------------
    combined = output + tables
    merge_boxes(combined)
    prepend_headings_to_text(combined)
    append_definition(combined)
    
    
    with open("data/processed/guideline_db.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)
        print(f"guideline_db.json file created with {len(combined)} chunks.")
    
    referenced_table_chunks = [chunk for chunk in combined if chunk['metadata']['type'] != 'paragraph']
    with open("data/processed/referenced_table_chunks.json", "w", encoding="utf-8") as f:
        json.dump(referenced_table_chunks, f, ensure_ascii=False, indent=4)
        
        
if __name__ == "__main__":
    main()