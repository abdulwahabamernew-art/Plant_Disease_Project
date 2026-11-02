from PIL import ImageDraw,ImageFont

def annotate(image, result):
    out=image.copy();draw=ImageDraw.Draw(out)
    font=ImageFont.load_default(size=max(12,min(20,image.width//60)))
    for leaf in result["leaves"]:
        review=bool(leaf.get("review_flags")) or result["plant_gate"]["label"]=="uncertain"
        color=(255,185,40) if review else (20,220,90)
        box=leaf["bbox"];draw.rectangle(box,outline=color,width=max(2,image.width//400))
        p=leaf["crop"]["top3"][0]
        label=f"{'REVIEW ' if review else 'Prediction '}#{leaf['id']} {p['class'].replace('___',' - ').replace('_',' ')} | {p['confidence']:.0%}"
        # Wrap long labels and keep text within the image.
        while draw.textbbox((0,0),label,font=font)[2] > image.width-8 and len(label)>20: label=label[:-6]+"..."
        tw,th=draw.textbbox((0,0),label,font=font)[2:]
        x=max(0,min(int(box[0]),image.width-tw-6));y=max(0,min(int(box[1])-th-8,image.height-th-8))
        draw.rectangle((x,y,x+tw+6,y+th+8),fill=(0,55,20));draw.text((x+3,y+2),label,fill="white",font=font)
    if not result["leaves"]:
        draw.text((8,8),f"Gate: {result['plant_gate']['label']} | leaves: 0",fill="red",font=font)
    else:
        text="EXPERIMENTAL predictions | disease not verified"
        while draw.textbbox((0,0),text,font=font)[2]>image.width-8 and len(text)>15:text=text[:-6]+"..."
        _,height=draw.textbbox((0,0),text,font=font)[2:]
        draw.rectangle((0,0,image.width,height+10),fill=(65,35,0))
        draw.text((4,4),text,fill="white",font=font)
    return out
