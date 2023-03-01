vid_to_cid = [[1], [1, 2, 3, 4], [23, 24, 25, 26], [5, 6], [8, 9], [22], [10, 11, 12], [0], [0], [14, 15], [16, 17], [18, 19], [20, 21]]
gen = (i+1 for i, lst in enumerate(vid_to_cid) if 20 in lst)
print(next(gen, -1))
print(next(gen, -1))
print(next(gen, -1))