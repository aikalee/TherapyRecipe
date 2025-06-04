import re
import json
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
def append_definition(match: re.Match) -> str:
    abbr = match.group(1)
    definition = abbr_map.get(abbr, "")
    return f"{abbr} ({definition})"

def find_graph_by_id(graphs, graph_id):
    """
    Find a graph by its ID in the graphs dictionary.
    """
    graph_id = graph_id.lower().replace(".", " ").strip().replace(" ", "_")
    print(f"Formatted graph ID: {graph_id}")
    for graph in graphs:
        if graph['metadata']['referee_id'] == graph_id:
            return graph['text']
        
    return None
        
with open("data/processed/graphs.json", "r", encoding="utf-8") as f:
    graphs = json.load(f)
    

from bs4 import BeautifulSoup, Tag, NavigableString

import sys
import re
import os
import json

chunk_id = 1

filename = "data/raw/source.html"
# if there's no input from command line, use the default filename
# if len(sys.argv) > 1:
    # filename = sys.argv[1]
    
with open(filename, "r", encoding="utf-8") as f:
    html = f.read()


# # I manually substitute the Level 1, Level 2, Level 3, Level 4 link with text like (Leve 1), (Level 2), (Level 3), (Level 4)
level1 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/62befe587468/10.1177_07067437241245384-img1.jpg'
level2 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/b9ea5ad77490/10.1177_07067437241245384-img2.jpg'
level3 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/5be38aafe33f/10.1177_07067437241245384-img3.jpg'
level4 = 'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/843a/11351064/68e56cd87632/10.1177_07067437241245384-img4.jpg'


soup = BeautifulSoup(html, "html.parser")
output = []


# parse the h1 title
title = soup.find("h1")
if title:
    title = title.decode_contents().replace('\n', '')
    
output.append({
    "text": title,
    "metadata":{
    "section": "title",
    "type": "title",
    "chunk_index": chunk_id,
    "headings": "Title",
    "referenced_tables": [],
    }
})

chunk_id += 1

countlevel1 = 0
countlevel2 = 0
countlevel3 = 0
countlevel4 = 0
# parse the main body
for p in soup.find_all("p"):
    referenced_tables = set()
    
    #-----------------------replace the <img> tags---------------------
    # we also manually delete the duplication in first occurrence mentioning Levels
    for img in p.find_all('img'):
        src = img.get('src')
        if src == level1:
            replacement_text = "(Level 1)"
            countlevel1 += 1
        elif src == level2:
            replacement_text = "(Level 2)"
            countlevel2 += 1
        elif src == level3:
            replacement_text = "(Level 3)"
            countlevel3 += 1
        elif src == level4:
            replacement_text = "(Level 4)"
            countlevel4 += 1
        else:
            continue    
            
        text_node = NavigableString(replacement_text)
        img.replace_with(text_node)
        referenced_tables.add('Table A')


        
        
    # ----------------------get section id----------------------------------
    parent_sec = p.find_parent(["section",'figure'], id=True)
    sec_id = parent_sec.get("id") if parent_sec else None
    
    #TODO: I'm currently ignoring the HTML tables, becuase AIKA's processing them.
    
    
    #-----------------------get the headings---------------------
    # special case: # manually finding the "No heading" in the html file to fix the No heading issue
    # delete the <div><\div> outside this to get the correct heading: <p>Protracted Discontinuation Symptoms and Hyperbolic Tapering Schedules.</p>
    
    # get the closest heading
    heading = p.find_previous_sibling(lambda tag: bool(re.match(r'^h[2-6]$', tag.name)))
    headings = heading.get_text(strip=True) if heading else 'No heading' 
    smallest_heading = headings
    
    if 'fig' in sec_id:
        referenced_tables.add(headings)
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
    # text = p.decode_contents().replace('\n', ' ') #get the text with <href> links and other tags
    
    
    #----------------------- mark Type of the text ------------------------
    type = 'paragraph'
    
    if 'table' in sec_id:  # either table image or table in HTML format
        if p.get('class') and 'img-box' in p.get('class'):
            type = 'table image '
        else: 
            continue
        # img_link = p.find('img').get('src')
        # print("img_link: ", img_link)
        # text = str(img_link)
        ##################################
        try:
            print(f"Finding graph by ID: {smallest_heading}")
            text = find_graph_by_id(graphs, smallest_heading)
        except Exception as e:
            print(f"Error finding graph by ID: {e}")
            continue
        ##################################

        # I only have 2 here with parent = <figure>, most of img's parent are <section id = 'table...'>
    elif p.get('class') and 'img-box' in p.get('class'):
        type = 'figure image'
        ##################################
        try:
            print(f"Finding graph by ID: {smallest_heading}")
            text = find_graph_by_id(graphs, smallest_heading)
        except Exception as e:
            print(f"Error finding graph by ID: {e}")
            continue
        ##################################
        # img_link = p.find('img').get('src')
        # print("img_link: ", img_link)
        # text = str(img_link)
        # sec_id = p.find_parent("figure", id=True).get("id")
    elif 'box' in sec_id: 
        type = 'box'
    
    
    
    #----------------------- get referenced tables ------------------------
    all_links = p.find_all('a')
    for link in all_links:
        href = link.get('href')
        if href.startswith('#'):
            referenced_tables.add(link.get_text(strip=True))
            
    # ----------------------- replace abbreviation with definition ------------------------
    # \b(ACT|ADHD|AI|…)\b
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in abbr_map.keys()) + r')\b'
    )
    text = pattern.sub(append_definition, text)
    
            
    #----------------------- formate the chunks ------------------------
    chunk = {
        "text": "From section: "+ headings + " > paragraph id: " + str(chunk_id) + "\n"+ text,
        "metadata": {
        "section": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11351064/#" + sec_id,
        "type": type,
        "chunk_index": chunk_id,
        "headings": headings,
        "referenced_tables": list(referenced_tables),
        }
    }
    
    output.append(chunk)
    
    chunk_id += 1
    
    
# print("count of levels: ", countlevel1, countlevel2, countlevel3, countlevel4)
    import json


with open('data/processed/tables.json', 'r', encoding='utf-8') as f2:
    tables = json.load(f2)

combined = output + tables

# ----------------------- write to json ------------------------
with open("data/guideline_db.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=4)
    print(f"guideline_db.json file created with {len(combined)} chunks.")