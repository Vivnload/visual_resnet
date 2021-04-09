from lxml import etree
from PIL import Image

xml=etree.parse('D:/mat/svt/svt1/train.xml')

root=xml.getroot()
image_dict={}
current_name=''
for article in root:
    for field in article:
        if field.tag == 'imageName':
            current_name=field.text
            image_dict[current_name]={}
            print(field.text)
        elif field.tag =='taggedRectangles':
            for i,tags in enumerate(field):
                print([tags.get('height'),tags.get('width'),tags.get('x'),tags.get('y')])
                image_dict[current_name][i]={}
                image_dict[current_name][i]['ax']=list(map(int,[tags.get('height'),tags.get('width'),tags.get('x'),tags.get('y')]))

                for tagg in tags:
                    image_dict[current_name][i]['label']=tagg.text
                    print(tagg.text)
    # break
print(image_dict)

for keys,values in image_dict.items():
    image = Image.open('D:/mat/svt/svt1/' + keys)
    for k,v in values.items():
        # for ks,vs in v.items():
        h,w,x,y=v['ax']
        image_crop=image.crop((x,y,x+w,y+h))
        image_crop.save('D:/mat/svt/svt1/train/' +str(k)+'_'+
                        str(h)+'_'+str(w)+'_'+str(x)+'_'+str(y)+'_'+
                        v['label']+'.jpg')

    #     image=Image.open('D:/mat/svt/'+key)
    #     for k,v in image_dict[key].items():
    #         for ks,vs in v.items()

