-- TO SEE INTERACTIVE USER INTERFACE GO TO https://naitrian.github.io/logistic_regression/ --

Basic logistic regression on the framingham dataset using pandas and numpy only.

Features scaled using Z score normalization and gradient descent done to minimize the cost (came down to 0.37)

The thing about working with medical projects is that the accuracy is not the sole thing that matters

initially i chose a "Sick" threshold of 0.5 i.e. that if the model predicted the chance of the person having coronary heart disease was 50% or more, it would consider it that the person will have a coronary heart disease

that made the accuracy 85% but the recall was 7%

that basically means for all sick people my model would only diagnose 7% of sick people as "sick"

so the threshold was changed manually until i found a super sweet spot at 0.15 (which i eventually narrowed down to 0.152)

then the funniest thing happened.

i got the accuracy (on the test dataset) 67%
and on the recall accuracy i also got a 67% lol

then the dust settled, i realized everything worked. W logistic regression.

anyways here are the final parameters:

Accuracy of Model : 67.57% 
Recall: 67.67%

Final Weights: 

male: 0.4142
age: 0.5631
education: -0.0088
currentSmoker: -0.1454
cigsPerDay: 0.3011
BPMeds: 0.2740
prevalentStroke: 0.7070
prevalentHyp: 0.3653
diabetes: -0.0959
totChol: 0.0701
sysBP: 0.2041
diaBP: -0.0210
BMI: -0.0078
heartRate: 0.0033
glucose: 0.2011

Final Bias: -2.19957

Standard Deviations:

age            8.548881
education      1.008107
cigsPerDay    11.880385
totChol       44.556093
sysBP         21.985431
diaBP         11.884921
BMI            4.111428
heartRate     12.090265
glucose       24.559457

Means:

age            49.580531
education       1.984366
cigsPerDay      8.900590
totChol       237.332743
sysBP         132.416224
diaBP          82.858702
BMI            25.809732
heartRate      76.105310
glucose        82.110324


