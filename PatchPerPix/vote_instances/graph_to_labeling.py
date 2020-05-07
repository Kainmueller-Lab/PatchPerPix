import logging
import numpy as np
import networkx as nx
from skimage.draw import line

if __package__ is None or __package__ == '':
    from aff_patch_graph import loadAffgraph
    from graph_mws import *
else:
    from .aff_patch_graph import loadAffgraph
    from .graph_mws import *


logger = logging.getLogger(__name__)

def affGraphToInstancesT(
        labels,
        patchshape,
        rad,
        debug_output1,
        debug_output2,
        instances,
        foreground_to_cover,
        affgraph, selected_patch_pairs,
        **kwargs
):
    affgraph = loadAffgraph(affgraph, selected_patch_pairs)

    return affGraphToInstances(affgraph, labels, patchshape, rad,
                               debug_output1, debug_output2,
                               instances, foreground_to_cover, **kwargs)


def affGraphToInstances(
        affinity_graph,
        labels,
        patchshape,
        rad,
        debug_output1,
        debug_output2,
        instances,
        foreground_to_cover,
        **kwargs
):
    logger.info("compute labeling")
    if kwargs['mws']:
        ccs = mws(affinity_graph)
    else:
        # if args.affinity_graph is None:
        rpgrph = nx.Graph()
        for e0, e1, a in affinity_graph.edges.data('aff'):
            if a > 0:
                rpgrph.add_edge(e0, e1, weight=a)
        # else:
        # rpgrph = affinity_graph
        ccs = nx.connected_components(rpgrph)

    # instance output:
    one_instance_per_channel = kwargs.get('one_instance_per_channel', False)
    instance_value = 0
    instance_list = []
    # cnt = 1
    for instance_value, cc in enumerate(ccs):
        if one_instance_per_channel:
            current_instance = np.zeros_like(instances)
        for idx in cc:
            if kwargs.get('sparse_labels'):
                patch = labels["_".join(str(i) for i in idx)]
            else:
                labelslice = tuple([slice(0, labels.shape[0])] +
                                   [idx[i] for i in range(len(idx))])
                patch = labels[labelslice]

            patch = np.reshape(patch, patchshape)
            start = idx - rad
            start = np.maximum(0, start)
            stop = idx + rad + 1
            stop = np.minimum(stop, instances.shape)
            startstopslice = tuple([slice(start[i], stop[i])
                                    for i in range(len(start))])
            if one_instance_per_channel:
                current_instance[startstopslice][
                    patch > kwargs['patch_threshold']] = instance_value + 1
            else:
                try:
                    instances[startstopslice][patch > kwargs['patch_threshold']] = instance_value + 1
                except:
                    print(start, stop, idx, patch.shape)
                # instances[startstopslice][patch > kwargs['patch_threshold']] = cnt
                # cnt += 1
                # if cnt == 256:
                #     cnt = 1

            if kwargs['debug']:
                debugstart = np.array(idx) * patchshape
                debugstop = debugstart + patchshape
                debugslice = tuple([slice(debugstart[i], debugstop[i])
                                    for i in range(len(debugstart))])
                debug_output1[debugslice] += instance_value + 1
        if one_instance_per_channel:
            instance_list.append(current_instance)
    if one_instance_per_channel:
        instances = np.stack(instance_list, axis=0)

    logger.info("done compute labeling")
    if kwargs['debug']:
        # todo: make debug_output1 work in 2d and 3d
        debug_output1[0:debug_output1.shape[0]:patchshape[0], :, :] = \
            instance_value + 2
        debug_output1[:, 0:debug_output1.shape[1]:patchshape[1], :] = \
            instance_value + 2
        debug_output1[:, :, 0:debug_output1.shape[2]:patchshape[2]] = \
            instance_value + 2

        for e0, e1, w in affinity_graph.edges.data('aff'):
            debugidx1 = np.array(e0) * patchshape + \
                        np.array(patchshape).astype(np.int16) / 2
            debugidx2 = np.array(e1) * patchshape + \
                        np.array(patchshape).astype(np.int16) / 2

            rr, cc = line(int(debugidx1[1]), int(debugidx1[2]),
                          int(debugidx2[1]), int(debugidx2[2]))
            debug_output2[0][rr, cc] = [max(d, float(w))
                                        if d != 0 else float(w)
                                        for d in debug_output2[0][rr, cc]]

        return instances, foreground_to_cover.astype(np.uint8), \
               debug_output1, debug_output2
    else:
        if kwargs.get("pad_with_ps", False):
            instances = instances[rad[0]:instances.shape[0]-rad[0],
                                  rad[1]:instances.shape[1]-rad[1],
                                  rad[2]:instances.shape[2]-rad[2]]
            foreground_to_cover = foreground_to_cover[
                rad[0]:foreground_to_cover.shape[0]-rad[0],
                rad[1]:foreground_to_cover.shape[1]-rad[1],
                rad[2]:foreground_to_cover.shape[2]-rad[2]]
        return instances, foreground_to_cover.astype(np.uint8)
