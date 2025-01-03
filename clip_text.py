# coding=utf-8
BACKGROUND_CATEGORY_VOC = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
                        ]

class_names_voc = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                   ]
                   

class_names_coco = ['person','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack',
                    'umbrella','handbag','tie','suitcase','frisbee',
                    'skis','snowboard','sports ball','kite','baseball bat',
                    'baseball glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','spoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor','laptop','mouse',
                    'remote','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hair drier','toothbrush',
]


BACKGROUND_CATEGORY_COCO = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge',
                        ]


class_names_nuswide = ['airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge', 'buildings', 'cars', 'castle', 'cat', 'cityscape', 'clouds', 'computer', 'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk', 'fire', 'fish', 'flags', 'flowers', 'food', 'fox', 'frost', 'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake', 'leaf', 'map', 'military', 'moon', 'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police', 'protest', 'railroad', 'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign', 'sky', 'snow', 'soccer', 'sports', 'statue', 'street', 'sun', 'sunset', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger', 'tower', 'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 'water', 'waterfall', 'wedding', 'whales', 'window', 'zebra']

class_names_cub = ['curved bill', 'dagger bill', 'hooked bill', 'needle bill', 'hooked_seabird bill', 'spatulate bill', 'all-purpose bill', 'cone bill', 'specialized bill', 'blue wing', 'brown wing', 'iridescent wing', 'purple wing', 'rufous wing', 'grey wing', 'yellow wing', 'olive wing', 'green wing', 'pink wing', 'orange wing', 'black wing', 'white wing', 'red wing', 'buff wing', 'blue upperparts', 'brown upperparts', 'iridescent upperparts', 'purple upperparts', 'rufous upperparts', 'grey upperparts', 'yellow upperparts', 'olive upperparts', 'green upperparts', 'pink upperparts', 'orange upperparts', 'black upperparts', 'white upperparts', 'red upperparts', 'buff upperparts', 'blue underparts', 'brown underparts', 'iridescent underparts', 'purple underparts', 'rufous underparts', 'grey underparts', 'yellow underparts', 'olive underparts', 'green underparts', 'pink underparts', 'orange underparts', 'black underparts', 'white underparts', 'red underparts', 'buff underparts', 'solid breast', 'spotted breast', 'striped breast', 'multi-colored breast', 'blue back', 'brown back', 'iridescent back', 'purple back', 'rufous back', 'grey back', 'yellow back', 'olive back', 'green back', 'pink back', 'orange back', 'black back', 'white back', 'red back', 'buff back', 'forked_tail tail', 'rounded_tail tail', 'notched_tail tail', 'fan-shaped_tail tail', 'pointed_tail tail', 'squared_tail tail', 'blue upper', 'brown upper', 'iridescent upper', 'purple upper', 'rufous upper', 'grey upper', 'yellow upper', 'olive upper', 'green upper', 'pink upper', 'orange upper', 'black upper', 'white upper', 'red upper', 'buff upper', 'spotted head', 'malar head', 'crested head', 'masked head', 'unique_pattern head', 'eyebrow head', 'eyering head', 'plain head', 'eyeline head', 'striped head', 'capped head', 'blue breast', 'brown breast', 'iridescent breast', 'purple breast', 'rufous breast', 'grey breast', 'yellow breast', 'olive breast', 'green breast', 'pink breast', 'orange breast', 'black breast', 'white breast', 'red breast', 'buff breast', 'blue throat', 'brown throat', 'iridescent throat', 'purple throat', 'rufous throat', 'grey throat', 'yellow throat', 'olive throat', 'green throat', 'pink throat', 'orange throat', 'black throat', 'white throat', 'red throat', 'buff throat', 'blue eye', 'brown eye', 'purple eye', 'rufous eye', 'grey eye', 'yellow eye', 'olive eye', 'green eye', 'pink eye', 'orange eye', 'black eye', 'white eye', 'red eye', 'buff eye', 'about_the_same_as_head bill', 'longer_than_head bill', 'shorter_than_head bill', 'blue forehead', 'brown forehead', 'iridescent forehead', 'purple forehead', 'rufous forehead', 'grey forehead', 'yellow forehead', 'olive forehead', 'green forehead', 'pink forehead', 'orange forehead', 'black forehead', 'white forehead', 'red forehead', 'buff forehead', 'blue under', 'brown under', 'iridescent under', 'purple under', 'rufous under', 'grey under', 'yellow under', 'olive under', 'green under', 'pink under', 'orange under', 'black under', 'white under', 'red under', 'buff under', 'blue nape', 'brown nape', 'iridescent nape', 'purple nape', 'rufous nape', 'grey nape', 'yellow nape', 'olive nape', 'green nape', 'pink nape', 'orange nape', 'black nape', 'white nape', 'red nape', 'buff nape', 'blue belly', 'brown belly', 'iridescent belly', 'purple belly', 'rufous belly', 'grey belly', 'yellow belly', 'olive belly', 'green belly', 'pink belly', 'orange belly', 'black belly', 'white belly', 'red belly', 'buff belly', 'rounded-wings wing', 'pointed-wings wing', 'broad-wings wing', 'tapered-wings wing', 'long-wings wing', 'large size', 'small size', 'very_large size', 'medium size', 'very_small size', 'upright-perching-water-like shape', 'chicken-like-marsh shape', 'long-legged-like shape', 'duck-like shape', 'owl-like shape', 'gull-like shape', 'hummingbird-like shape', 'pigeon-like shape', 'tree-clinging-like shape', 'hawk-like shape', 'sandpiper-like shape', 'upland-ground-like shape', 'swallow-like shape', 'perching-like shape', 'solid back', 'spotted back', 'striped back', 'multi-colored back', 'solid tail', 'spotted tail', 'striped tail', 'multi-colored tail', 'solid belly', 'spotted belly', 'striped belly', 'multi-colored belly', 'blue primary', 'brown primary', 'iridescent primary', 'purple primary', 'rufous primary', 'grey primary', 'yellow primary', 'olive primary', 'green primary', 'pink primary', 'orange primary', 'black primary', 'white primary', 'red primary', 'buff primary', 'blue leg', 'brown leg', 'iridescent leg', 'purple leg', 'rufous leg', 'grey leg', 'yellow leg', 'olive leg', 'green leg', 'pink leg', 'orange leg', 'black leg', 'white leg', 'red leg', 'buff leg', 'blue bill', 'brown bill', 'iridescent bill', 'purple bill', 'rufous bill', 'grey bill', 'yellow bill', 'olive bill', 'green bill', 'pink bill', 'orange bill', 'black bill', 'white bill', 'red bill', 'buff bill', 'blue crown', 'brown crown', 'iridescent crown', 'purple crown', 'rufous crown', 'grey crown', 'yellow crown', 'olive crown', 'green crown', 'pink crown', 'orange crown', 'black crown', 'white crown', 'red crown', 'buff crown', 'solid wing', 'spotted wing', 'striped wing', 'multi-colored wing']


class_names_LaSO = ['bicycle', 'boat', 'stop sign', 'bird', 'backpack', 'frisbee', 'snowboard', 'surfboard', 'cup', 'fork', 'spoon', 'broccoli', 'chair', 'keyboard', 'microwave', 'vase']