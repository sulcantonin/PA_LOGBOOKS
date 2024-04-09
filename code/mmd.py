from tqdm import tqdm
import re, sys
sys.path.insert(0,'/home/sulcan/Documents/lora_nips/code/')
from tex import *
from collections import defaultdict

def replace_displayed_math_mmd_to_latex(paper : str) -> str:
    # replace \[.*\] with equation mode $$
    eq_expr = re.compile(r'\\\[(.*?)\\\]', flags = re.DOTALL)
    while len(re.findall(eq_expr, paper)) != 0:
        eq_re = re.search(eq_expr,paper)
        paper = paper[:eq_re.start()] + \
            '$$' + paper[eq_re.start() + 2 :eq_re.end()-2] + '$$' + \
            paper[eq_re.end():]
    return paper


def replace_math_mmd_to_latex(paper : str) -> str:
    # replace inline math \(.*\) with $ $
    eq_expr = re.compile(r'\\\((.*?)\\\)', flags = re.DOTALL)
    while len(re.findall(eq_expr, paper)) != 0:
        eq_re = re.search(eq_expr,paper)
        paper = paper[:eq_re.start()] + \
            '$' + paper[eq_re.start() + 2 :eq_re.end()-2] + '$' + \
            paper[eq_re.end():]
    return paper

def prepare_mmd_eqations_and_tables_for_simcse(data_mmd : dict):
    for k in tqdm(data_mmd):
        paper = data_mmd[k]

        paper = replace_displayed_math_mmd_to_latex(paper)
        paper = replace_math_mmd_to_latex(paper)

        # replace tables with empty string!
        tabular_expr = tag_expr_full('table')
        while len(re.findall(tabular_expr,paper)):
            tabular_re = re.search(tabular_expr,paper)
            try:
                tabular_plain = ''
                # tabular_plain = pypandoc.convert_text(tabular_re.group(), 'markdown',format = 'latex', extra_args=['--wrap=none'])
            except Exception as e:
                tabular_plain = ""
                # print(f'EXCEPTION {str(e)} {tabular_re.group()}')
            paper = paper.replace(tabular_re.group(), tabular_plain)

        data_mmd[k] = paper
    return data_mmd

def prepare_mmd_eqations_for_pacuna(data_mmd : dict) -> dict:
    for k in tqdm(data_mmd):
        paper = data_mmd[k]
        paper = replace_displayed_math_mmd_to_latex(paper)
        paper = replace_math_mmd_to_latex(paper)
        data_mmd[k] = paper
    return data_mmd

def extract_latex_tables(data_mmd : dict) -> dict:
    tables = defaultdict(lambda : [])

    table_cap_re = re.compile(r'^\s*Table \d+:(.*)\n')
    table_re = tag_expr_full('table')
    for k in tqdm(list(data_mmd.keys())):
        paper = data_mmd[k]
        abstract = find_abstract(paper)

        while table := re.search(table_re,paper):
            end = table.end()
            caption = ''
            # looks if the part of the table (irrespectively of how many white space
            # fills the space contains Table \d:
            table_cap = re.search(table_cap_re,paper[table.end():])
            if table_cap:
                caption = table_cap.group()
                end += table_cap.end()

            tables[k].append({'table_with_caption' : paper[table.start():end],
                              'table' : table.group(),
                              'caption' : caption,
                              'abstract' : abstract})

            paper = re.sub(re.escape(paper[table.start():end]), '---', paper)
        #TODO: replacement
    return tables

def find_abstract(paper : str, no_abstract = '') -> str:
    '''
    function returns an abstract, returns the whole section which must 
    - either begin with #+abstrct 
    - or be _abstract_
    
    if no rule is met, returns @no_abstract 
    '''
    
    abstract_title_re = r'^#+\s*abstract'
    
    paper = re.sub(r'_abstract_', r'# abstract', paper)
    paper = paper.split('\n#')
    pars = [par for par in paper if re.findall(abstract_title_re, par, re.IGNORECASE)]
    if len(pars) == 0: # no abstract
        return no_abstract 
    else:
        return pars[0]
    
    
