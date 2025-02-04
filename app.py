from fastai.learner import load_learner
import gradio

learner = load_learner('models/model.pth')

def classify_image(img):
    categories = ('Cheeseburger', 'Hotdog')
    pred, idx, probs = learner.predict(img)
    return dict(zip(categories, map(float,probs)))

def main():
    image = gradio.Image(width=192, height=192)
    label = gradio.Label()
    examples = ['examples/hotdog.jpg', 'examples/cheeseburger.jpg', 'examples/pizza.jpg'] 
    interface = gradio.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
    interface.launch(inline=False)

if __name__ == '__main__':
    main()
