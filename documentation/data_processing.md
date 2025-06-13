## How we processed data

1. save html page
2. Text parsing:
  - delete head and tail of the file. Start from title and delete the sections after ## conclusion(from supplementary materials). Delete the french abstract
  - Observe and identify the link of each section
  - extract h1/h2/h3 tag to identify the hierarchy of the section in article
  - get headings, concatnate all the headings together and pre-pend it to the actual paragraph texts.
  - Abbreviation substitution. Detect all the abbreviation in the text and replace them with full definition.
  - detect what tables/figures are mentioned in the texts
4. Analyze special element in the texts: table image and HTML Table:
    - table image
        - table embedded in the page is accessed from link. Get the links form HTML source page.
        - grab the link and send it to Vision model(llama-11B one). convert the table to natural language format to describe the table. There are many duplications for convenience of the model's understanding. 
          - we observe a lot of green circles pictures that represents level1, level2 and 3,4. These pics are saved as link in the the html. We search for the links and then replace them with plain text.
          - get table name and store it in `referee_id`
    - html table
        - detect the html structure and parse it manually with some custom python code
        - get table name and store it in `referee_id`


    - manually merge caption under the figure
    - some paragraphs in the Box is separated in different chunks. merged them into one chunk. The longest of them is around 250 words(estimated around 350 tokens, length ok)

# As we explore the data, design schema of the json structure database

{
  "text": "The results are summarized in Table 2...",
  "embedding": [...],
  "metadata": {
    "section": "Results",
    "type": "paragraph",
    "chunk_id": 42,
    "headings": "Abstract/Background",
    "referenced_tables": ["table_2"],
    "referee_id": "table_2"
  }
}