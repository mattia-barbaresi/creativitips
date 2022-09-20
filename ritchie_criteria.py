import bz2


def compute_ncd(a, b):
    ca = float(len(bz2.compress(a)))
    cb = float(len(bz2.compress(b)))
    cab = float(len(bz2.compress(a + b)))
    return (cab - min(ca, cb)) / max(ca, cb)


def create_string(vector):
    result = ""
    for x in vector:
        result = result + "".join(x)
    return result


def calculate_ratio(x, y):
    return len(x) / len(y)


def calculate_av_typ(x, repertoire):
    sum = 0
    for element in x:
        sum = sum + calculate_typicality(element, repertoire)
    return sum / len(x)


def calculate_typicality(results, repertoire):
    results_string = create_string(results)
    repertoire_string = create_string(repertoire)
    ncd = compute_ncd(results_string, repertoire_string)
    return ncd


def compute_criterion_1(results, string_repertoire):
    n = 0
    avg_eval = 0
    for x in results:
        if x != "":  # avoid empty strings
            res = compute_ncd("".join(x), string_repertoire)
            avg_eval = avg_eval + res
            n = n + 1
    return avg_eval / n


def compute_criterion_2(results, string_repertoire, alpha):
    n = 0
    count = 0
    for x in results:
        if x != "":  # avoid empty strings
            res = compute_ncd("".join(x), string_repertoire)
            if res > alpha:
                count = count + 1
            n = n + 1
    return count * 1.0 / n


def compute_criterion_2_edited(results, string_repertoire, higher_bound, lower_bound):
    n = 0
    count = 0
    for x in results:
        if x != "":  # avoid empty strings
            res = compute_ncd("".join(x), string_repertoire)
            if res > higher_bound or res < lower_bound:
                count = count + 1
            n = n + 1
    return count * 1.0 / n
