import os
import json

def generate_txt():
    """
    generate path for dataloader
    """
    if not os.path.exists(os.path.join('.', 'txt', 'train')):
        os.makedirs(os.path.join('.','txt','train'))
    if not os.path.exists(os.path.join('.','txt','test')):
        os.makedirs(os.path.join('.','txt','test'))
    for i in range(2,25):
        with open('./MEIP_Micro_FACS_Codes.json', encoding='utf-8') as f:
            t = json.load(f)
            for j in t:
                if int(t[j]['Subject'])==i:
                    with open(os.path.join('.','txt','test',str(i)+'.txt'),'a') as f:
                        label=str_label_to_int_label(t[j]['reallabel'],t[j]['masklabel'],3)
                        # if len(t[j]["ApexFrame"])==2:
                        #     image_name='00'+t[j]["ApexFrame"]+'.jpg'
                        # if len(t[j]["ApexFrame"])==1:
                        #     image_name='000'+t[j]["ApexFrame"]+'.jpg'
                        # if len(t[j]["ApexFrame"])==3:
                        #     image_name='0'+t[j]["ApexFrame"]+'.jpg'
                        # if len(t[j]["ApexFrame"])==4:
                        #     image_name=t[j]["ApexFrame"]+'.jpg'
                        emotion_label1=emotion_label(t[j]['reallabel'])
                        image_path=os.path.join('.','data','MEIP_Onset_Apex_frame_30_new',t[j]['Filename'])
                        f.write(image_path+' '+label+' '+emotion_label1+'\n')
                else:
                    with open(os.path.join('.','txt','train',str(i)+'.txt'),'a') as f1:
                        label = str_label_to_int_label(t[j]['reallabel'], t[j]['masklabel'],3)
                        # if len(t[j]["ApexFrame"])==2:
                        #     image_name='00'+t[j]["ApexFrame"]+'.jpg'
                        # if len(t[j]["ApexFrame"])==1:
                        #     image_name='000'+t[j]["ApexFrame"]+'.jpg'
                        # if len(t[j]["ApexFrame"])==3:
                        #     image_name='0'+t[j]["ApexFrame"]+'.jpg'
                        # if len(t[j]["ApexFrame"])==4:
                        #     image_name=t[j]["ApexFrame"]+'.jpg'
                        emotion_label1 = emotion_label(t[j]['reallabel'])
                        image_path = os.path.join('.', 'data', 'MEIP_Onset_Apex_frame_30_new', t[j]['Filename'])
                        f1.write(image_path + ' ' + label + ' '+emotion_label1+ '\n')


def emotion_label(str):
    if str == 'an':
        label = '0'
    elif str == 'di':
        label = '1'
    elif str == 'fe':
        label = '2'
    elif str == 'ha':
        label = '3'
    elif str == 'sa':
        label = '4'
    elif str == 'su':
        label = '5'
    return label
def str_label_to_int_label(true_str,mask_str,mode=1):
    """

    :param true_str: 真实表情的label字符
    :param mask_str: 伪装表情的label字符
    :param mode: 1.返回真实标签 2.返回伪装标签 3.返回36分类标签
    :return:
    """

    if mode==1:
        str=true_str
        if str == 'an':
            label = '0'
        elif str == 'di':
            label = '1'
        elif str == 'fe':
            label = '2'
        elif str == 'ha':
            label = '3'
        elif str == 'sa':
            label = '4'
        elif str == 'su':
            label = '5'
    if mode==2:
        str=mask_str
        if str == 'an':
            label = '0'
        elif str == 'di':
            label = '1'
        elif str == 'fe':
            label = '2'
        elif str == 'ha':
            label = '3'
        elif str == 'sa':
            label = '4'
        elif str == 'su':
            label = '5'
    if mode ==3:
        # ------------------------------an---------------------------
        if true_str == 'an' and mask_str == 'an':
            label = '0'
        elif true_str == 'an' and mask_str == 'fe':
            label = '1'
        elif true_str == 'an' and mask_str == 'ha':
            label = '2'
        elif true_str == 'an' and mask_str == 'su':
            label = '3'
        elif true_str == 'an' and mask_str == 'sa':
            label = '4'
        elif true_str == 'an' and mask_str == 'di':
            label = '5'
        # ------------------------------di---------------------------
        if true_str == 'di' and mask_str == 'sa':
            label = '6'
        elif true_str == 'di' and mask_str == 'an':
            label = '7'
        elif true_str == 'di' and mask_str == 'di':
            label = '8'
        elif true_str == 'di' and mask_str == 'fe':
            label = '9'
        elif true_str == 'di' and mask_str == 'su':
            label = '10'
        elif true_str == 'di' and mask_str == 'ha':
            label = '11'

        # ------------------------------fe---------------------------
        if true_str == 'fe' and mask_str == 'sa':
            label = '12'
        elif true_str == 'fe' and mask_str == 'ha':
            label = '13'
        elif true_str == 'fe' and mask_str == 'an':
            label = '14'
        elif true_str == 'fe' and mask_str == 'su':
            label = '15'
        elif true_str == 'fe' and mask_str == 'fe':
            label = '16'
        elif true_str == 'fe' and mask_str == 'di':
            label = '17'
        # ------------------------------ha---------------------------
        if true_str == 'ha' and mask_str == 'sa':
            label = '18'
        elif true_str == 'ha' and mask_str == 'an':
            label = '19'
        elif true_str == 'ha' and mask_str == 'ha':
            label = '20'
        elif true_str == 'ha' and mask_str == 'su':
            label = '21'
        elif true_str == 'ha' and mask_str == 'fe':
            label = '22'
        elif true_str == 'ha' and mask_str == 'di':
            label = '23'
        # ------------------------------sa---------------------------
        if true_str == 'sa' and mask_str == 'sa':
            label = '24'
        elif true_str == 'sa' and mask_str == 'an':
            label = '25'
        elif true_str == 'sa' and mask_str == 'su':
            label = '26'
        elif true_str == 'sa' and mask_str == 'ha':
            label = '27'
        elif true_str == 'sa' and mask_str == 'di':
            label = '28'
        elif true_str == 'sa' and mask_str == 'fe':
            label = '29'
        # ------------------------------su---------------------------
        if true_str == 'su' and mask_str == 'sa':
            label = '30'
        elif true_str == 'su' and mask_str == 'an':
            label = '31'
        elif true_str == 'su' and mask_str == 'ha':
            label = '32'
        elif true_str == 'su' and mask_str == 'su':
            label = '33'
        elif true_str == 'su' and mask_str == 'fe':
            label = '34'
        elif true_str == 'su' and mask_str == 'di':
            label = '35'
    return label
generate_txt()

