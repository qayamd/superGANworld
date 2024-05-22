# superGANworld

## What is it?
A set of tools for training a Generative adversarial network to generate super mario world levels. \
This project is reliant on FuSoYa's [Lunar Magic](https://fusoya.eludevisibility.org/lm/) level editor, as well as it's .mwl file standard \
It is important to note that I have not included the model or any of the training dataset, as it was a collection of fanmade custom levels from the rom hack community [SMWCentral](https://www.smwcentral.net/). 
I believe that the non-consentual usage of other people's works (copyrighted or not) as training data for AI is unethical.
This project does not contain any of Nintendo's code, assets, or intellectual property. 

## What does what?
* gan.py: Contains the training and optimization code for the model
* query.py: Calls the model from the local directory, and generates a level in .txt form
* toMWL.py: Converts a .txt file formatted as the output of toTXT.py and converts it back to a valid lunar magic level
* toTxt.py: Converts a lunar magic .mwl file to a specifically formatted .txt file

## What do I get?
This repo does not contain any level data or model of any kind. However, when trained on a sufficient dataset, it should produce quality results. Some example sections:
![Local image](/images/1.png)
![Local image](/images/2.png)
Clearly, some fine tuning is still required. It is important to note the lack of conceptual understanding on the part of the model of what a "good" level would look like.
These images are examples from a GAN trained on ~2000 super mario world levels.

## What's to come?
I originally had the idea that this would work with an open source emulator like [snes9x](https://www.snes9x.com/), which would run the game, while the GAN would generate levels
on the fly, which would then be injected in the game's memory during runtime. However, other projects have captured my interest, and fine tuning the GAN and 
developing a deep enough understanding of Super Mario World's inner workings seem incredibly time consuming. I might return to this one day.
