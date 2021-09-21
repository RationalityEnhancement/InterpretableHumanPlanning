from formula_visualization import dnf2conj_list as d2l, unit_pred2expr
from hyperparams import CLEANED_PREDS as condied_preds
import re

def pred_dictt(pred, one_step=True):
    """
    High-level funciton to pass appropriate arguments for expression building
    functions.
    """
    not_ = False
    if 'not' == pred[:3] :
        if 'not(among' in pred: pred = pred[4:-1]
        not_ = True
    if pred[-1] == ')': pred= pred[:-1] + ' )'
    if pred[-2:] == ') ': pred = pred[:-2] + ' )'
    o = logic2words(pred, one_step=one_step)
    if not_: out = o.lower()
    else: out = o
    return(out)
    
def unit_pred2expr(predicate_str, unit=False, first_conj=False,
                   second_conj=False, special=False, one_step=True, until=False):
    """
    Convert a string being a predicate to an expression.
    
    For SIMPLE and GENERAL predicates from RL2DT.PLP.DSL
    """
    ## SIMPLE predicates
    if ' not' in predicate_str or 'not ' in predicate_str:
        real_pred = predicate_str[5:-1]
        not_ = 'not '
    elif ' not ' in predicate_str:
        real_pred = predicate_str[6:-1]
        not_ = 'not '
    elif 'not' in predicate_str:
        real_pred = predicate_str[4:-1]
        not_ = 'not '
    else:
        real_pred = predicate_str
        not_ = ''

    if re.match('\s{0,1}depth\(', real_pred):
        x = real_pred.split()[1]
        if x[-1] == ')': x = x[:-1]
        if x[-1] == ' ': x = x[:-1]
        conj = ''
        on = 'on'
        if not_ != '': 
            if one_step:
                add = ' an arbitrary level but for '
            else:
                add = ' arbitrary levels but for '
        else:
            add = ' level '
        if second_conj:
            conj = ' and '
        if first_conj or second_conj or unit:
            if one_step:
                on = 'a node on'
            else:
                on = 'nodes on'
        ## 'that reside'+s+
        return conj + on + add + x

    if 'on_highest' in real_pred:
        s = 's'
        if not_ != '':
            not_ = 'does not '
            not_m = 'do not '
            s = ''
        if not(one_step):
            s = ''
        if first_conj or unit:
            if one_step:
                return 'a node that '+not_+'belong'+s+' to the best path'
            else:
                return "nodes that " + not_m + 'belong to the best path'
        if second_conj:
            if one_step:
                return 'and '+not_+'belong'+s+' to the best path'
            else:
                return 'and '+not_+'belong to the best path'
        if special:
            return 'that '+not_+'belong'+s+' to the best path'
        return 'that '+not_+'belong'+s+' to the best path'
    if '2highest' in real_pred:
        s = 's'
        if not_ != '':
            not_ = 'does not '
            not_m = 'do not '
            s = ''
        if not(one_step):
            s = ''
        if first_conj or unit:
            if one_step:
                return 'a node that '+not_+'belong'+s+' to the second-best path'
            else:
                return "nodes that " + not_m + 'belong to the second-best path'
        if second_conj:
            if one_step:
                return 'and '+not_+'belong'+s+' to the second-best path'
            else:
                return 'and '+not_+'belong to the second-best path'
        if special:
            return 'that '+not_+'belong'+s+' to the second-best path'
        return 'that '+not_+'belong'+s+' to the second-best path'
        
    if 'are_branch' in real_pred:
        if not_ != '':
            message = 'belonging to a subtree with some unobserved leaves'
        else:
            message = 'belonging to a subtree with all leaves already observed'
        if first_conj or unit:
            if one_step:
                return 'a node ' + message
            else:
                return "nodes " + message
        if second_conj:
            return ' and ' + message
        if special:
            return ' ' + message
        return message

    if 'is_leaf' in real_pred:
        if not_ != '':
            not_ = 'non-'
        if first_conj or unit:
            if one_step:
                return 'a ' + not_ + 'leaf'
            else:
                return not_ + 'leaves'
        if second_conj or special:
            if one_step:
                return not_ + 'leaf'
            else:
                return not_ + 'leaves'       
        return not_ + 'leaves'
    if 'is_root' in real_pred:
        if not_ != '':
            not_ = 'non-'
        if first_conj or unit:
            if one_step:
                return 'a ' + not_ + 'root'
            else:
                return not_ + 'roots'
        if second_conj or special:
            if one_step:
                return not_ + 'root'
            else:
                return not_ + 'roots' 
        return not_ + 'roots'
    if 'is_max_in' in real_pred:
        nodes = 'a node '
        whose = ' its '
        if not(one_step):
            nodes = 'nodes '
            whose = ' their '
        if second_conj or special:
            nodes = ''
        if not_ != '':
            return nodes + 'with no 48 on'+whose+'path'
        return nodes + 'with a 48 on'+whose+'path'
    if 'is_2max_in' in real_pred:
        nodes = 'a node '
        whose = ' it belongs '
        if not(one_step):
            nodes = 'nodes '
            whose = ' they belong '
        if second_conj or special:
            nodes = ''
        if not_ != '':
            return nodes + 'with both the paths'+whose+'to containing a 48'
        return nodes + 'with one of the paths it'+whose+'to not containing a 48'
    if 'is_child' in real_pred:
        if not_ == '':
            which = 'positive'
        else:
            which = 'negative or unobserved'
        if first_conj or unit:
            return 'a node that has a child with a '+which+' value'
        if second_conj:
            return 'and has a child with a '+which+' value'
        if special:
            if one_step:
                return 'that has a child with a '+which+' value'
            return 'that have children with a '+which+' value'
        return 'that have children with a '+which+' value'
    if 'is_parent' in real_pred:
        if not_ == '':
            which = 'positive'
        else:
            which = 'negative or unobserved'
        if first_conj or unit:
            if one_step:
                return 'a node that has a parent with a '+which+' value'
            else:
                return 'nodes that have a parent with a '+which+' value'
        if second_conj:
            if one_step:
                return 'and has a parent with a '+which+' value'
            else:
                return 'and have a parent with a '+which+' value'
        if special:
            if one_step:
                return 'that has a parent with a '+which+' value'
            return 'that have parents with a '+which+' value'
        return 'that have parents with a '+which+' value'
    if 'highest' in real_pred or 'smallest' in real_pred:
        which = 'highest' if 'highest' in real_pred else 'lowest'
        if 'leaf' in real_pred:
            which = '48' if 'highest' in real_pred else '-48'
            if not_ != '':
                finish = 'different from ' + which
            else:
                finish = which
            if first_conj or unit:
                if one_step:
                    return 'a node that leads to a leaf whose value is ' + finish
                else:
                    return 'nodes that lead to leaves whose value is ' + finish
            if second_conj:
                if one_step:
                    return 'and that leads to a leaf whose value is ' + finish
                else:
                    return 'and that lead to leaves whose values is ' + finish
            if special:
                if one_step:
                    return 'that leads to a leaf whose value is ' + finish
                return 'that lead to leaves whose value is ' + finish
            return 'that lead to leaves whose value is ' + finish
        if 'root' in real_pred:
            which = '4' if 'highest' in real_pred else '-4'
            if not_ != '':
                finish = 'different from ' + which
            else:
                finish = which
            if first_conj or unit:
                if one_step:
                    return 'a node accessible by a root whose value is ' + finish
                else:
                    return 'nodes accessible by roots whose value is ' + finish
            if second_conj:
                if one_step:
                    return 'accessible by a root whose value is ' + finish
                else:
                    return 'accessible by roots whose value is ' + finish
            if special:
                if one_step:
                    return 'accessible by a root whose value is ' + finish
                return 'accessible by a root whose value is ' + finish
            return 'accessible by a root whose value is ' + finish
        if 'child' in real_pred:
            if not_ != '':
                not_ = 'non-'
            else:
                not_ = ''
            if first_conj or unit:
                if one_step:
                    return 'a node that has a child with the ' + not_ + which \
                           + ' value on its level'
                else:
                    return 'nodes that have children with the ' + not_ + which \
                           + ' value on their level'
            if second_conj:
                if one_step:
                    return ' and has a child with the ' + not_ + which \
                           + ' value on its level'
                else:
                    return ' and have children with the ' + not_ + which \
                           + ' value on their level'
            if special:
                if one_step:
                    return 'that has a child with the ' + not_ + which \
                           + ' value on its level'
                else:
                    return 'that have children with the ' + not_ + which \
                           + ' value on their level' 
            return 'that have children with the ' + not_ + which \
                   + ' value on their level'
        if 'parent' in real_pred:
            if not_ != '':
                not_ = 'non-' + which + ' or unobserved'
            else:
                not_ = which
            if first_conj or unit:
                if one_step:
                    return 'a node that has a parent with the ' + not_ \
                           + ' value on its level'
                else:
                    return 'nodes that have parents with the ' + not_ \
                           + ' value on their level'
            if second_conj:
                if one_step:
                    return 'and has a parent with the ' + not_ \
                           + ' value on its level'
                else:
                    return 'and have parents with the ' + not_ \
                           + ' value on their level'
            if special:
                if one_step:
                    return 'that has a parent with the ' + not_ \
                           + ' value on its level'
                else:
                    return 'that have parents with the ' + not_ \
                           + ' value on their level' 
            return 'nodes that have parents with the '+ not_ \
                   + ' value on their level'

    if 'is_observed' in real_pred:
        nodes = ''
        if not_ != '':
            not_ = 'un'
        if one_step:
            an = 'an '
        else:
            an = ''
        if unit:
            if one_step:
                nodes = ' node'
            else:
                nodes = ' nodes'
        return an + not_ + 'observed' + nodes

    ## GENERAL predicates
    if 'is_none' in real_pred:
        return 'nothing was observed yet'
    if 'is_all' in real_pred:
        return 'all the nodes are observed'
    if 'is_path' in real_pred:
        if not_ != '':
            return 'no path is fully observed'
        return 'there exists an observed path'
    if 'is_parent_depth' in real_pred:
        if not_ != '':
            return 'some of the nodes one level above are unobserved'
        return 'all the nodes one level above are observed'
    if 'is_previous_observed' in real_pred:
        ed = 'ed' if not(until) else 's'
        e = 'the previously observed node uncover' + ed
        if 'positive' in real_pred:
            
            if not_ != '':
                return e + ' a negative value'
            return e+ ' a positive value'
        if 'parent' in real_pred:
            whose = 'its' if one_step else 'their'
            if not_ != '':
                return 'the previously observed node was not '+whose+' parent'
            return 'the previously observed node was '+whose+' parent'
        if 'sibling' in real_pred:
            whose = 'its' if one_step else 'their'
            if not_ != '':
                return 'the previously observed node was not '+whose+' sibling'
            return 'the previously observed node was '+whose+' sibling'
        if 'max_level' in real_pred:
            w = real_pred.split()
            num = w[1]
            if num[-1] == ')': num = num[:-1]
            if num == str(3):
                if not_ != '':
                    return e + ' something else than a 48'
                return e + ' a 48'
            if num == str(2):
                if not_ != '':
                    return e + ' something else than an 8'
                return e + ' an 8'
            if num == str(1):
                if not_ != '':
                    return e + ' something else than a 4'
                return e + ' a 4'
            if not_ != '':
                return e + ' a non-maximum value on level ' + num
            return e + ' the maximum value on level ' + num
        if 'min_level' in real_pred:
            w = real_pred.split()
            num = w[1]
            if num[-1] == ')': num = num[:-1]
            if num == str(3):
                if not_ != '':
                    return e + ' something else than a -48'
                return e + ' a -48'
            if num == str(2):
                if not_ != '':
                    return e + ' something else than a -8'
                return e + ' a -8'
            if num == str(1):
                if not_ != '':
                    return e + ' something else than a -4'
                return e + ' a -4'
            if not_ != '':
                return e + ' a non-minimum value on level ' + num
            return ' the minimum value on level ' \
                   + num
        if 'max_nonleaf' in real_pred:
            if not_ != '':
                return e + ' something else than an 8'
            return e + ' an 8'
        if 'max_root' in real_pred:
            if not_ != '':
                return e + ' something else than a 4'
            return e + ' a 4'
        if 'max' in real_pred:
            if not_ != '':
                return e + ' something else than a 48'
            return e + ' a 48'
        if 'min' in real_pred:
            if not_ != '':
                return e + ' something else than a -48'
            return e + ' a -48'
    if 'is_positive' in real_pred:
        if not_ != '':
            return 'neither of the observed nodes has a positive value'
        return 'a node with a positive value is observed'
    if 'are_leaves' in real_pred:
        if not_ != '':
            return 'some of the leaves are unobserved'
        return 'all the leaves are observed'
    if 'are_roots' in real_pred:
        if not_ != '':
            return 'some of the roots are unobserved'
        return 'all the roots are observed'
    if 'previous_observed_depth' in real_pred:
        w = real_pred.split()
        num = w[1]
        if num[-1] == ')': num = num[:-1]
        if not_ != '':
            return 'the previously observed node was not at level ' + num
        return 'the previously observed node was at level ' + num
    if 'count' in real_pred:
        w = real_pred.split()
        num = w[1]
        if num[-1] == ')': num = num[:-1]
        if num == 1:
            if not_ != '':
                return 'nothing is observed yet'
            else:
                return 'there is at least 1 observed node'
        if not_ != '':
            return 'there are less than ' + num + ' observed nodes'
        return 'there are at least ' + num + ' observed nodes'
    if 'termination' in real_pred:
        w = real_pred.split()
        num = w[1]
        if num[-1] == ')': num = num[:-1]
        if not_ != '':
            return 'the termination reward is not ' + num
        return 'the termination reward is ' + num

    return predicate_str
    
def pred2expr(predicate_str, unit=False, conjunction=False, one_step=False, until=False):
    """
    Join expressions of two SIMPLE predicates or output an expression for one, if
    there is only one.
    """
    if ' or ' in predicate_str:
        p1, p2 = predicate_str.split(' or ')
        p1 = p1[1:]
        p2 = p2[:-1]
        expr = unit_pred2expr(p1, first_conj=True, until=until) \
                   + ' or ' + unit_pred2expr(p2,second_conj=True, until=until)
    elif ' and ' in predicate_str:
        p1, p2 = predicate_str.split(' and ')
        if p1 == p2:
            expr = unit_pred2expr(p1, unit=True, one_step=one_step)
        elif 'is_observed' in p1: 
            if 'is_root' in p2 or 'is_leaf' in p2:
                expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) \
                       + ' ' + unit_pred2expr(p2, special=True, one_step=one_step)
            else:
                node = ' node ' if one_step else ' nodes '
                expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) \
                       + node + unit_pred2expr(p2, special=True, one_step=one_step)
        elif 'is_observed' in p2: 
            if 'is_root' in p1 or 'is_leaf' in p1:
                expr = unit_pred2expr(p2, first_conj=True, one_step=one_step) \
                       + ' ' + unit_pred2expr(p1, special=True, one_step=one_step)
            else:
                node = ' node ' if one_step else ' nodes '
                expr = unit_pred2expr(p2, first_conj=True, one_step=one_step) \
                       + node + unit_pred2expr(p1, special=True, one_step=one_step)
        elif 'is_root' in p1 or 'is_leaf' in p1 or 'depth' in p1:
            con = ' '
            if 'is_root' in p2 or 'is_leaf' in p2 or 'depth' in p2:
                con = ' and '
            expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) + con \
                   + unit_pred2expr(p2, special=True, one_step=one_step)
        elif 'is_root' in p2 or 'is_leaf' in p2 or 'depth' in p2:
            con = ' '
            if 'is_root' in p1 or 'is_leaf' in p1 or 'depth' in p1:
                con = ' and '
            expr = unit_pred2expr(p2, first_conj=True, one_step=one_step) + con \
                   + unit_pred2expr(p1, special=True, one_step=one_step)
        elif not conjunction:
            expr = unit_pred2expr(p1, one_step=one_step,unit=True) + ' and ' \
                   + unit_pred2expr(p2, one_step=one_step, unit=True)
        else:
            expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) \
                   + ' ' + unit_pred2expr(p2,second_conj=True, one_step=one_step)
    else:
        expr = unit_pred2expr(predicate_str, one_step=one_step, until=until, unit=True)
    return expr
    
def among_pred2expr(predicate_str, prev, all_=False, one_step=True):
    """
    Convert a string being a predicate to an expression.
    
    For AMONG/ALL predicates from RL2DT.PLP.DSL
    """
    add = ' nodes described by the point above'
    add = pred2expr(prev, one_step=False)
    if 'has_smallest' in predicate_str:
        if all_:
            return 'are on the same level'
        if one_step:
            return '\n - it is located on the lowest level considering the' + add
        return '\n - they are located on the lowest level considering the' + add
    if 'has_largest' in predicate_str:
        if all_:
            return 'are on the same level'
        if one_step:
            return '\n - it is located on the highest level considering the' + add
        return '\n - they are located on the highest level considering the' + add

    if 'has_best_path' in predicate_str:
        if all_:
            return 'lie on best paths'    
        if one_step:
            return "\n - lies on a best path"
        return "\n - lie on best paths"
    
    expr = predicate_str
    if 'has_child_highest' in predicate_str:
        if all_:
            if one_step:
                expr = "\n - it has a child with the highest value " + \
                       "considering the children of other " + add
            else:
                expr = "\n - they have children with the highest values " + \
                       "considering the children of other " + add
        if all_:
            return 'have a child with the same observed value'
    if 'has_child_smallest' in predicate_str:
        if one_step:
            expr = "\n - it has a child with the lowest value considering " + \
                   "the children of other " + add
        else:
            expr = '\n - they have children with the lowest values '+ \
                   'considering the children of other ' + add
        if all_:
            return 'have a child with the same observed value'
    if 'has_parent_highest' in predicate_str:
        if one_step:
            expr = "\n - it has a parent with the highest value considering " + \
                   "the parents of other " + add
        else:
            expr = "\n - they have parents with the highest values " + \
                   "considering the parents of other " + add
        if all_:
            return 'have a parent with the same observed value'
    if 'has_parent_smallest' in predicate_str:
        if one_step:
            expr = "\n - it has a parent with the lowest value considering " + \
                   "the parents of other " + add
        else:
            expr = "\n - they have parents with the lowest values " + \
                   "considering the the parents of other " + add
        if all_:
            return 'have a parent with the same observed value'
    if 'has_leaf_highest' in predicate_str:
        if one_step:
            expr = "\n - it leads to a leaf with the highest value " + \
                   "considering the leaves accessible by other " + add
        else:
            expr = "\n - they lead to leaves with the highest values " + \
                   "considering the leaves accessible by other " + add
        if all_:
            return 'lead to a leaf with the same observed value'
    if 'has_leaf_smallest' in predicate_str:
        if one_step:
            expr = "\n - it leads to a leaf with the lowest value " + \
                   "considering the leaves accessible by other " + add
        else:
            expr = "\n - they lead to leaves with the lowest values " + \
                   "considering the leaves accessible by other " + add
        if all_:
            return 'lead to a leaf with the same observed value'
    if 'has_root_highest' in predicate_str:
        if one_step:
            expr = "\n - it is accessible by a root with the highest value " + \
                   "considering the roots leading to other " + add
        else:
            expr = "\n - they are accessible by roots with the highest value " + \
                   "considering the roots leading to other " + add
        if all_:
            return 'are accessible by a root with the same observed value'
    if 'has_root_smallest' in predicate_str:
        if one_step:
            expr = "\n - it is accessible by a root with the lowest value " + \
                   "considering the roots leading to other " + add
        else:
            expr = "\n - they are accessible by roots with the lowest value " + \
                   "considering the roots leading to other " + add
        if all_:
            return 'are accessible by a root with the same observed value'
    return expr
    
def logic2words(pred, one_step=True):
    """
    Convert a string being a predicate to an expression.
    
    For COMPSITIONAL predicates from PLP.DSL
    """
    if re.match('among', pred):
        beg = 'it is ' if one_step else "they are "
        if re.search(':', pred):
            among_pred = re.search(':.+ \)', pred).group(0)[2:-2]
            list_preds = re.search('.+(?= :)', pred[6:]).group(0)
            what = pred2expr(list_preds, one_step=one_step)
            expr = '\n - ' + beg + what + among_pred2expr(among_pred, 
                                                           list_preds, 
                                                           one_step=one_step)
                
        else:
            list_preds = pred[6:-2]
            expr = '\n - ' + beg + pred2expr(list_preds, 
                                              conjunction=True, 
                                              one_step=one_step)
            
    elif re.match('all_', pred) or re.match('not\(all_', pred):
        among_pred = re.search(':.+ \)', pred).group(0)[2:-2]
        list_preds = re.search('.+(?= :)', pred[5:]).group(0)
        if 'not(' == pred[:4]:
            beg = 'only part of '
        else:
            beg = 'all '
        expr = beg + 'the ' + pred2expr(list_preds, one_step=False) + ' ' + \
               among_pred2expr(among_pred, list_preds, all_=True)

    elif not(any([pred in c for c in condied_preds])):
        beg = 'it is ' if one_step else "they are "
        expr = '\n - ' + beg + unit_pred2expr(pred, unit=True, one_step=one_step)
    else:
        expr = unit_pred2expr(pred, unit=True, one_step=one_step)

    return expr
                  
def order_dnf(dnf_list):
    """
    Order a list of predicates starting from positive ones, then predicates that
    contain the preicate 'all_', and finally negated predicates.
    """
    dos, donts, alls = [], [], []
    for pred in dnf_list:
        if any([pred in cp for cp in condied_preds]) or 'all_' is pred:
            alls.append(pred)
        elif pred[:3] == 'not':
            donts.append(pred)
        else:
            dos.append(pred)
    return dos + alls + donts
    
def dnf2command(dnf_string, one_step=False):
    """
    Turn a DNF formula into a command in natural language using functions that 
    translate predicates into small expressions depending on the structure of
    the DNF.
    """
    preds = d2l(dnf_string,paran=False)[0]
    preds = order_dnf(preds)
    full = ''
    num_all, num_click, num_dont = 0,0,0

    for num, sub_pred in enumerate(preds):
        nodes = 'a node' if one_step else 'the nodes'
                
        if any([sub_pred in cp for cp in condied_preds]) or 'all_' in sub_pred:
            if num_click == 0 and num_all == 0 and num_dont <= 1:
                full += "Click on random nodes. "
            num_all += 1
            if num_all > 1: beg = ''
            else:
                if one_step:
                    beg = '\n\nClick in this way under the condition that:'
                else: 
                    beg = '\n\nClick in this way as long as:'
            full += beg + '\n - '+ pred_dictt(sub_pred, one_step=one_step)
            if not(any([sub_pred in cp for cp in condied_preds])): full += '. '
            
        elif sub_pred[:3] == 'not':
            num_dont += 1
            next = pred_dictt(sub_pred, one_step=one_step)
            if num_click == 0 and num_all == 0 and num_dont == 1:
                full += "Click on random nodes. "
            if full[-2:] != '. ': full += '. '
            if num_dont > 1:
                full += next
            else:
                full += '\n\nDo not click on '+ nodes + \
                        ' satisfying either of the following conditions: ' + next

        else:
            num_click += 1
            if 'True' in sub_pred:
                if one_step:
                    full += 'Click on a random node or terminate. '
                else:
                    full += 'Terminate or click on some random nodes and then terminate. '
            else:
                next = pred_dictt(sub_pred, one_step=one_step)
                if num_click > 1:
                    full += next
                else:
                    full += 'Click on '+ nodes + \
                            ' satisfying all of the following conditions:' + next
    if full[-2:] != '. ': full += '.' 
    return full
    
def procedure2text(procedure_str):
    """
    Transform a procedural description/formula into a natural language text.
    
    Treat each DNF before AND NEXT as a step with number i, with i=1 for the
    first DNF, etc.
    
    Add appropriate suffixes to the comand if the DNF is followed by UNTIL or
    UNLESS.
    """
    if 'False' in procedure_str:
        txt = 'Do not click.'
        print("\n\n        Whole instructions: \n\n{}".format(txt))
        return txt
    if 'None' in procedure_str:
        txt = 'NO INSTRUCTION FOUND. (Treated as the no-planning strategy).'
        print("\n\n        Whole instructions: \n\n{}".format(txt))
        return txt
    txt = ''
    split = procedure_str.split('\n\nLOOP FROM ')
    if len(split) == 2:
        body, go_to = split
    else:
        body, go_to = split[0], None
    dnfs = body.split(' AND NEXT ')
    id_dict = {}
    for i, dnf in enumerate(dnfs):
        split_dnf = dnf.split(' UNTIL ')
        split_dnf2 = dnf.split(' UNLESS ')
        txt += str(i+1) + '. '
        if len(split_dnf) == 2 and len(split_dnf2) == 1:
            dnf_cond, until = split_dnf
            txt += dnf2command(dnf_cond)
            until_cond = 'until ' + pred2expr(until, until=True) if 'IT' not in until \
                         else "as long as possible"
            txt += "\n\nRepeat this step " + until_cond + ".\n\n(s)"
        elif len(split_dnf) == 1 and len(split_dnf2) == 1:
            dnf_cond = split_dnf[0]
            txt += dnf2command(dnf_cond, one_step=True) + "\n\n(s)"
        elif len(split_dnf) == 1 and len(split_dnf2) == 2:
            dnf_cond, unless = split_dnf2
            txt += 'Unless ' + pred2expr(unless, until=True) + \
                   ', in which case stop at the previous step, '
            txt += dnf2command(dnf_cond, one_step=True).lower() + '\n\n(s)'
        else:
            dnf_cond, untill = split_dnf
            until, unless = untill.split(' UNLESS ')
            until_cond = 'until ' + pred2expr(until, until=True) if 'IT' not in until \
                         else "as long as possible"
            txt += dnf2command(dnf_cond) + ". "
            txt += "\n\nRepeat this step " + until_cond + " unless " + \
                   pred2expr(unless, until=True)+' -- then stop at the previous step.\n\n(s)'
        id_dict[dnf_cond] = i    
            
    if go_to != None:
        goto_split = go_to.split(' UNLESS ')
        if len(goto_split) == 2:
            go_to, unless = goto_split
        else:
            go_to, unless = goto_split[0], None
        txt += str(len(dnfs)+1) + ". GOTO step " + str(id_dict[go_to]+1)
        if unless != None:
            txt += ' unless ' + pred2expr(unless, until=True)
        txt += '.'
    print("\n\n        Whole instructions: \n\n{}".format(txt))
    return txt
    
def alternative_procedures2text(procedures_str):
    """
    Separate alternative procedural formulas and create a natural language
    descriptions for each.
    """
    procs = procedures_str.split('\n\nOR\n\n')
    out = ''
    for n, p in enumerate(procs):
        if n>0: out += '\n\nOR\n\n'
        out += aligned_print(procedure2text(p))
    print("\n\n\n          FINAL INSTRUCTIONS: \n\n{}".format(out))
    return out
    
def aligned_print(txt):
    """
    Format text so that it took 80 lines and then new line was started.
    
    Changed to fit the particular description and some tabulation was added
    for arrows ->, etc.
    """
    new_txt = ''
    steps = txt.split('\n\n(s)')
    counter = 80
    for step in steps:
        counter = 80
        arrow = False
        for num, s in enumerate(step):
            new_txt += s
            if num+1 < len(step) and s+step[num+1] =='\n ':
                arrow = True
                new_txt += '  '
                counter = 78
            elif s == '\n':
                arrow = False
                new_txt += '   '
                counter = 77
            counter -= 1
            if counter <= 0 and (num+1 >= len(step) or step[num+1] == ' ') and arrow:
                new_txt += '\n     '
                counter = 74
            elif counter <= 0 and (num+1 >= len(step) or step[num+1] == ' '):
                new_txt += '\n  '
                counter = 77
        new_txt += '\n\n'
    return(new_txt)
