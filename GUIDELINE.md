# **How to contribute?**

Instruction:
1. Install [Git](https://git-scm.com/) or [GitHub Desktop](https://desktop.github.com/), and an editor such as Visual Studio Code.

2. Clone the codes to your computer (**qsee** is a core package of our various repository)
```
git clone https://github.com/vutuanhai237/UC-VQA.git
cd UC-VQA
rm -rf qsee (delete folder qsee)
git clone https://github.com/vutuanhai237/qsee.git
```

Note that the **qsee** folder must be on the same level as the **codes** folder.

3. Make sure that you have installed python 3+ version. After that, install all needed packages.
```
pip install -r requirements.txt
```
4. Test

Run all test case
```
cd tests
pytest
```