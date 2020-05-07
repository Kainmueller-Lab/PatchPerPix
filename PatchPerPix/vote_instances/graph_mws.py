import logging
import numpy as np


logger = logging.getLogger(__name__)

def mws(affgraph):
    logger.info("compute mws")
    edges = []
    nodes = {}
    node_CCs = {}
    node_IDs = {}
    mutex = set()
    ccs = {}
    for i, n in enumerate(affgraph.nodes()):
        nodes[i] = n
        node_CCs[i] = 0
        node_IDs[n] = i

    for e0, e1, a in affgraph.edges.data('aff'):
        if a > 0:
            edges.append((node_IDs[e0], node_IDs[e1], a, 1))
        else:
            edges.append((node_IDs[e0], node_IDs[e1], -a, -1))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)

    ccs[0] = set(node_IDs.values())
    numedges = len(edges)

    for i, (e0, e1, a, isattr) in enumerate(edges):
        if i % 100 == 0:
            logger.info("mws: edge %s of %s", i, numedges)

        if isattr == 1 and not (e0, e1) in mutex:
            if node_CCs[e0] == 0 and node_CCs[e1] == 0:
                new_CC = np.max(list(node_CCs.values())) + 1
                ccs[new_CC] = set([e0, e1])
                ccs[0].discard(e0)
                ccs[0].discard(e1)
                node_CCs[e0] = new_CC
                node_CCs[e1] = new_CC
            # try merging, then check for mutex of merged CCs:
            elif node_CCs[e0] == 0 or node_CCs[e1] == 0:
                CC = max(node_CCs[e0], node_CCs[e1])
                ena = e0 if node_CCs[e0] == 0 else e1
                has_mutex = np.any([(node_CCs[e] == CC and f == ena) or
                                    (node_CCs[f] == CC and e == ena)
                                    for (e, f) in mutex])
                if not has_mutex:
                    trycc = ccs[CC] | set([e0, e1])
                    ccs[CC] = trycc
                    ccs[0].discard(e0)
                    ccs[0].discard(e1)
                    node_CCs[e0] = CC
                    node_CCs[e1] = CC
            elif node_CCs[e0] != node_CCs[e1]:
                CC0 = node_CCs[e0]
                CC1 = node_CCs[e1]
                has_mutex = np.any([
                    (node_CCs[e] == CC0 and node_CCs[f] == CC1) or
                    (node_CCs[f] == CC0 and node_CCs[e] == CC1)
                    for (e, f) in mutex])
                if not has_mutex:
                    trycc = ccs[node_CCs[e0]] | ccs[node_CCs[e1]]
                    CC = min(node_CCs[e0], node_CCs[e1])
                    remove_CC = max(node_CCs[e0], node_CCs[e1])
                    ccs[CC] = trycc
                    for e in ccs[remove_CC]:
                        node_CCs[e] = CC
                    ccs[remove_CC] = set()
            if False:
                logger.info("adding edge %s %s %s %s", e0, e1, a, isattr)
                logger.info("ccs %s", ccs)
                logger.info(" mutex %s", mutex)
        else:
            mutex.add((e0, e1))

    ccs_output = []
    for cc in ccs.keys():
        if cc > 0:
            ccs_output.append([nodes[i] for i in ccs[cc]])

    logger.info("done compute mws")
    return ccs_output
