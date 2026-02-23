import numpy as np
import pandas as pd


def shelf_merge(stats,s1,s2,snew):
    stats["labels"] = list(stats["labels"])
    s1 = stats["labels"].index(s1)
    s2 = stats["labels"].index(s2)
    if s2>s1:
        s1, s2 = s2, s1

    stats["labels"].append(snew)

    s1area = stats["areas"][s1]
    s2area = stats["areas"][s2]

    s1gllen = stats["gllen"][s1]
    s2gllen = stats["gllen"][s2]

    for k in stats.keys():
        if k != "labels":
            print(k)
            stats[k] = list(stats[k])

            if k in ['cdws','salts','raw_temps','Tcdw','gprimes',\
                     'fs-1','gldepths','entrance_thickness','front_thick','front_depth']:
                stats[k].append((stats[k][s1]*s1gllen + stats[k][s2]*s2gllen)\
                                /(s1gllen + s2gllen)) 

            elif k in ['slopes','mys','sigmas','avg_drafts']:
                stats[k].append((stats[k][s1]*s1area + stats[k][s2]*s2area)\
                                /(s1area + s2area)) 

            elif k in ['areas','Bmelt','Bpolyna',\
                    'Btotal']:
                stats[k].append(stats[k][s1] + stats[k][s2])

            elif k == 'shelf_class':
                stats[k].append(stats[k][s1] and stats[k][s2])

            elif k == 'shelf_color':
                stats[k].append(stats[k][s1])

            stats[k].pop(s1)
            stats[k].pop(s2)

            stats[k] = np.asarray(stats[k])

    stats["labels"].pop(s1)
    stats["labels"].pop(s2)

    stats["labels"] = np.asarray(stats["labels"])

    return stats
                
def read_shelf_class(labels):
    shelf_class = pd.read_csv("shelf_classification.csv",sep=',')
    shelf_color = []
    shelf_classnumber = []
    for i in labels:
        classification = shelf_class.loc[shelf_class['Shelf Name']==i].values[0][1]
        explanation = shelf_class.loc[shelf_class['Shelf Name']==i].values[0][3]
        if classification and type(classification) == str:
            if 'both' in classification:
                shelf_color.append("gray")
                shelf_classnumber.append(0)
            if 'disconnected' in classification:
                if type(explanation) == str:
                    shelf_color.append("plum")
                    shelf_classnumber.append(-0.5)
                else:
                    shelf_color.append("fuchsia")
                    shelf_classnumber.append(-1)
            elif 'connected' in classification:
                if type(explanation) == str:
                    shelf_color.append("bisque")
                    shelf_classnumber.append(0.5)
                else:
                    shelf_color.append("orange")
                    shelf_classnumber.append(1)
            elif 'unknown' in classification:
                shelf_color.append("gray")
                shelf_classnumber.append(0)
            else:
                1+1
                # print(i)
                # print(classification)
        else:
            shelf_color.append('white')
            shelf_classnumber.append(np.nan)
    return shelf_classnumber,shelf_color
 
