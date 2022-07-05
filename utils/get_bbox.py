import xlrd

def get_bbox_from_image(img_name, typ):
    xl = xlrd.open_workbook('/home/abc/datasets/PolyDataset2/position_label.xlt')
    xl1 = xlrd.open_workbook('/home/abc/datasets/PolyDataset/position_label.xlt')
    if typ == 'wht':
        table = xl.sheets()[0]
        table1 = xl1.sheets()[0]
    else:
        table = xl.sheets()[1]
        table1 = xl1.sheets()[1]

    col = table.col_values(0)
    key = '\''+img_name+'\''
    if key in col:
        ind = col.index(key)
    else:
        col = table1.col_values(0)
        if key not in col:
            key = img_name
        ind = col.index(key)
        table = table1

    pos = table.cell(ind, 1).value[1:-1]
    pos = list(map(lambda x: int(x), pos.split(',')))
    pos = [pos[1], pos[0], pos[3], pos[2]]
    return pos

if __name__ == '__main__':
    bbox = get_bbox_from_image("181-101.png", 'wht')
    print(bbox)
