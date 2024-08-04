from model.summarizer import Summarizer

def main():
    text = """
     Mobile Phone is often also called “cellular phone”. It is a device mainly used for a voice call. Presently technological advancements have made our life easy. Today, with the help of a mobile phone we can easily talk or video chat with anyone across the globe by just moving our fingers. Today mobile phones are available in various shapes and sizes, having different technical specifications and are used for a number of purposes like – voice calling, video chatting, text messaging or SMS, multimedia messaging, internet browsing, email, video games, and photography. Hence it is called a ‘Smart Phone’. Like every device, the mobile phone also has its pros and cons which we shall discuss now.
    """
    
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    print("Original Text:\n", text)
    print("\nSummary:\n", summary)

if __name__ == "__main__":
    main()
