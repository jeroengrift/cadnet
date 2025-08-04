import cv2
import torch
import numpy as np
import torch.nn.functional as F

from scipy.spatial.distance import cdist

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th]

# adjusted with only junc prediction, not concave and convex
def get_pred_junctions(junc_map, joff_map):
    junc_pred_nms = non_maximum_suppression(junc_map)
    topK = min(300, int((junc_pred_nms>0.008).float().sum().item()))
    juncs_pred, _ = get_junctions(junc_map, joff_map, topk=topK)

    return juncs_pred.detach().cpu().numpy()

def ext_c_to_poly_coco(ext_c, im_h, im_w):
    mask = np.zeros([im_h+1, im_w+1], dtype=np.uint8)
    polygon = np.int0(ext_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_prop_mask[f_y + 1, f_x] = 1
    trans_prop_mask[f_y, f_x + 1] = 1
    trans_prop_mask[f_y + 1, f_x + 1] = 1
    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly

def diagonal_to_square(poly):
    new_c = []
    for id, p in enumerate(poly[:-1]):
        if (p[0] + 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]) \
                or (p[0] - 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
        elif (p[0] + 1 == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] + 1, p[1]])
        elif (p[0] - 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] - 1, p[1]])
        elif (p[0] + 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0], p[1] - 1])
        else:
            new_c.append(p)
            new_c.append([p[0], p[1] + 1])
    new_poly = np.asarray(new_c)
    new_poly = np.concatenate((new_poly, new_poly[0].reshape(-1, 2)))
    return new_poly

def inn_c_to_poly_coco(inn_c, im_h, im_w):
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(inn_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_prop_mask[f_y[np.where(f_y == min(f_y))], f_x[np.where(f_y == min(f_y))]] = 0
    trans_prop_mask[f_y[np.where(f_x == min(f_x))], f_x[np.where(f_x == min(f_x))]] = 0
    #trans_prop_mask[max(f_y), max(f_x)] = 1
    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)[::-1]
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    #return poly
    new_poly = diagonal_to_square(poly)
    return new_poly

def simple_polygon(poly, thres=10):
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:,1], vec0[:,0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:,1], vec1[:,0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)

    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1,2)))
    return simple_poly

def generate_polygon(prop, mask, juncs, pid):
    poly, score, juncs_pred_index = get_poly_crowdai(prop, mask, juncs)
    juncs_sa, edges_sa, junc_index = get_junc_edge_id(poly, juncs_pred_index)
    return poly, juncs_sa, edges_sa, score, juncs_pred_index

def generate_polyline(prop, mask, juncs, pid):
    poly, score, juncs_pred_index = get_poly_crowdai(prop, mask, juncs)
    juncs_sa, edges_sa, junc_index = get_junc_edge_id(poly, juncs_pred_index)
    return poly, juncs_sa, edges_sa, score, juncs_pred_index

def get_poly_crowdai(prop, mask_pred, junctions):
    prop_mask = np.zeros_like(mask_pred).astype(np.uint8)
    prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
    masked_instance = np.ma.masked_array(mask_pred, mask=(prop_mask != 1))
    score = masked_instance.mean()
    im_h, im_w = mask_pred.shape
    contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly = []
    edge_index = []
    jid = 0
    for contour, h in zip(contours, hierarchy[0]):
        c = []
        if h[3] == -1:
            c = ext_c_to_poly_coco(contour, im_h, im_w)
        if h[3] != -1:
            if cv2.contourArea(contour) >= 50:
                c = inn_c_to_poly_coco(contour, im_h, im_w)
            #c = inn_c_to_poly_coco(contour, im_h, im_w)
        if len(c) > 3:
            init_poly = c.copy()
            if len(junctions) > 0:
                cj_match_ = np.argmin(cdist(c, junctions), axis=1)
                cj_dis = cdist(c, junctions)[np.arange(len(cj_match_)), cj_match_]
                u, ind = np.unique(cj_match_[cj_dis < 5], return_index=True)
                if len(u) > 2:
                    ppoly = junctions[u[np.argsort(ind)]]
                    ppoly = np.concatenate((ppoly, ppoly[0].reshape(-1, 2)))
                    init_poly = ppoly
            init_poly = simple_polygon(init_poly, thres=10)
            poly.extend(init_poly.tolist())
            edge_index.append([i for i in range(jid, jid+len(init_poly)-1)])
            jid += len(init_poly)
    return np.array(poly), score, edge_index

def get_junc_edge_id(poly, junc_pred_index):
    juncs_proj = []
    junc_index = []
    edges_sa = []
    jid = 0
    for ci, cc in enumerate(junc_pred_index):
        juncs_proj.extend(poly[cc])
        if ci == 0:
            junc_id = np.asarray(cc).reshape(-1, 1)
            jid += len(junc_id)
        else:
            junc_id = np.arange(jid, jid+len(cc)).reshape(-1, 1)
            jid += len(cc)
        edge_id = np.concatenate((junc_id, np.roll(junc_id, -1)), axis=1)
        edges_sa.extend(edge_id)
        junc_index.append(junc_id.ravel())
    edges_sa = np.asarray(edges_sa)
    juncs_proj = np.asarray(juncs_proj)
    return juncs_proj, edges_sa, junc_index

def junc_edge_to_poly(juncs, junc_pred_index):
    poly = []
    for ci, cc in enumerate(junc_pred_index):
        if ci == 0:
            poly.extend(juncs[cc].tolist())
            poly.append(juncs[0].tolist())
        else:
            if len(cc) > 2:
                poly.extend(juncs[np.asarray(cc)-1].tolist())
                poly.append(juncs[cc[0]-1].tolist())
    poly = np.asarray(poly)
    return poly
