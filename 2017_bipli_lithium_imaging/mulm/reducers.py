# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:10:31 2013

@author: jinpeng.li@cea.fr
"""

import numpy as np
from epac.map_reduce.reducers import Reducer
from epac.map_reduce.results import Result
from epac.workflow.base import key_pop


class PValR2Reducer(Reducer):
    """Reducer that computes p-values of stattistics.

    """
    def __init__(self):
        self.pattern = "Perm\(nb=([a-zA-Z0-9]+)(.*)"

    def get_diff_perm_nbs(self, result):
        import re
        import numpy as np
        res_keys = result.keys()
        ret_diff_perm_nbs = set()
        for res_key in res_keys:
            re_res = re.search(self.pattern, res_key)
            if re_res:
                ret_diff_perm_nbs.add(int(re_res.group(1)))
        return list(ret_diff_perm_nbs)

    def get_max_r2_with_perm_nb(self, result, perm_nb):
        import re
        import numpy as np
        res_keys = result.keys()
        ret_max_r2 = None
        for res_key in res_keys:
            re_res = re.search(self.pattern, res_key)
            if re_res:
                perm_nb_i = re_res.group(1)
                if perm_nb_i == repr(perm_nb):
                    cur_max = np.max(result[res_key]['r2'])
                    if not ret_max_r2:
                        ret_max_r2 = cur_max
                    elif ret_max_r2 < cur_max:
                        ret_max_r2 = cur_max
        return ret_max_r2

    def reduce(self, result):
        diff_perm_nbs = self.get_diff_perm_nbs(result)
        max_r2 = {}
        for perm_nb in diff_perm_nbs:
            max_r2[perm_nb] = self.get_max_r2_with_perm_nb(result,
                                                           perm_nb)
        r2_no_perms = max_r2[0]
        count = 0
        for i in max_r2:
            if i == 0:
                continue
            if r2_no_perms < max_r2[i]:
                count += 1
        p_value = float(count) / float(len(max_r2))
        _, res_key = key_pop(result.keys()[0], index=-1)
        out = Result(key=res_key)
        out["pval"] = p_value
        return out
