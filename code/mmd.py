from tqdm import tqdm
import re, sys
sys.path.insert(0,'/home/sulcan/Documents/lora_nips/code/')
from tex import *


def prepare_mmd_eqations_and_tables_for_simcse(data_mmd : dict):
    for k in tqdm(data_mmd):
        x = data_mmd[k]

        # replace \[.*\] with equation mode $$
        eq_expr = re.compile(r'\\\[(.*?)\\\]', flags = re.DOTALL)
        while len(re.findall(eq_expr, x)) != 0:
            eq_re = re.search(eq_expr,x)
            x = x[:eq_re.start()] + '$$' + x[eq_re.start() + 2 :eq_re.end()-2] + '$$' + x[eq_re.end():]


        # replace inline math \(.*\) with $ $
        eq_expr = re.compile(r'\\\((.*?)\\\)', flags = re.DOTALL)
        while len(re.findall(eq_expr, x)) != 0:
            eq_re = re.search(eq_expr,x)
            x = x[:eq_re.start()] + '$' + x[eq_re.start() + 2 :eq_re.end()-2] + '$' + x[eq_re.end():]

        # replace tables
        tabular_expr = tag_expr_full('table')
        while len(re.findall(tabular_expr,x)):
            tabular_re = re.search(tabular_expr,x)
            try:
                tabular_plain = ''
                # tabular_plain = pypandoc.convert_text(tabular_re.group(), 'markdown',format = 'latex', extra_args=['--wrap=none'])
            except Exception as e:
                tabular_plain = ""
                # print(f'EXCEPTION {str(e)} {tabular_re.group()}')
            x = x.replace(tabular_re.group(), tabular_plain)

        data_mmd[k] = x
    return data_mmd