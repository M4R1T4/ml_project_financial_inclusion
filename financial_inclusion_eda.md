# Financial Inclusion in Africa

![](/plots_and_pictures/africa_symbol_big.png)

In the years 2016-2018, survey was conducted in Kenya, Tanzania, Rwanda and Uganda to find out how many people in these countries have a bank account (or not) and other factors that could have an influence.


* 23,500 people were interviewed 
* 14% of the people use a bank account
* 86% have no access to a bank account


## The Countries

The survey was conducted in 4 East African countries:  
 Kenya, Tanzania, Rwanda and Uganda.   

![overview_africa](/plots_and_pictures/Africa_Overview.png)

Although these 4 are neighbouring countries, they differ greatly in terms of their populations, landscapes, historical conflicts and the resulting economic structures.   

The two largest countries are Tanzania and Kenya. Both border the Indian Ocean to the east. On the western side, both, together with Uganda, border Lake Victoria, the largest freshwater lake in Africa. Rwanda, on the other hand, is the smallest country of the four. It also differs from the others in that it is very hilly and forms the border between the Nile and Congo rivers. 
We also expect to see differences in the data between the individual countries. 


## Comparison between Population and Number of respondents

We compared the number of respondents in the respective countries with the population of these countries at the time of the survey. The aim was to see whether the ratio is the same. 

![population_vs_number](/plots_and_pictures/population_vs_number.png)

On the left-hand side of the chart, the bars show the number of people in the countries during the survey period. The bars on the right show the number of people surveyed in the respective country. As can be seen, the ratio of people interviewed in the different countries does not reflect the ratio of the population between the countries. For example, most people were interviewed in Rwanda, although Rwanda has by far the smallest population of the four countries. 

## Bank Accounts

The Pie chart shows the percentage of those who have a bank account in yellow and those who don’t have a account in green.

![Bank Account Rate](/plots_and_pictures/bank_account_rvrsd.png)

Only 14 percent of the people interviewed have a bank account at all. That is very few and we will have to take this imbalance into account in the course of the project.

In the data analysis we will often show the bank account rate to compare different classes or countries.   

When modelling the prediction model, we are faced with the challenge that the larger class will dominate the predictions. To compensate for this, we will try out various methods. 


## Bank Account Rates per Country

It is also interesting to see whether the bank account rate is the same in all countries or whether there are differences. For this reason, we have analysed the percentage of bank accounts for the individual countries.   
Here too, the pie chart shows the proportion of participants without bank accounts in green and the proportion with bank accounts per country in yellow. 

![Bank Account Rate per country](/plots_and_pictures/bank_account_countrys.png)

The rate is very different in the various countries.
A quarter of respondents in Kenya have a bank account.
But in the other three countries, the proportion of people with bank access is much smaller (11 % in Rwanda, 9% in Uganda and Tanzania)

## Gender

An other question is: Who is more likely to have a bank account, Men or Woman?  

For this analysis, we first presented the proportion of women and men and then divided this up to show the proportion of bank accounts per gender.

![gender](/plots_and_pictures/gender.png)


As we see, the majority of respondents are women (around 60%). 
But only 11 % of female interviewees have a bank account, thats near to every . 

In contrast, almost every fifth man (19%) has a bank account. The ratio between the two genders is almost 2:1. 

## Cell Phone Access

Respondents were asked whether or not they had access to a cell phone. The answers resulted in different groups with regard to bank account and mobile phone access.

![Cell_Phone_vs_Bank_account](/plots_and_pictures/CellPhone_vs_BankAccount.png)

The Matrix shows 4 groups:

* People who have both: a bank account and access to a cell phone  (13.6%)
* People who have a bank account but no access to a cell phone (0.4%)
* People who have neither a bank account nor access to a cell phone (25.4%)
* People who do not have a bank account but do have access to a cell phone (60.6%)  

The last group is the most interesting. It is also the largest. In this group, the respondents have a mobile phone but no bank account. They may already be using mobile payment options anyway. This group would be the ideal target group for banks to reach via straightforward mobile app access. 

## Education Level

Respondents were asked about their highest educational qualification. We have analysed this information in different ways. 

### Education Level per Country

As we are interested in seeing whether there are differences between the educational qualifications in the individual countries, we have broken these down by country.   
Because the number of people surveyed varied from country to country, we have calculated the rate of educational level in relation to the whole country. The bars therefore show the proportion of respondents from a country who declared the respective level of education. 

![Education_per_Country](/plots_and_pictures/Education_level_per_Country.png)

We can see that the majority of respondents had at least a primary education. The proportion then decreases with the higher levels of education. In Tanzania and Rwanda, there is also a large proportion with no education, whereas Tanzania also has the highest proportion of higher education. Uganda and Kenya are relatively similar in the distribution of educational attainment. But the other two countries are more different from the two and from each other. 

### Dependence between Education level and Bank Account

As we only have 14% bank accounts, we wanted to see which education level has the highest proportion of bank accounts. In this graph you can see on the left side the number of respondents who have this education level. On the right side you can see the bank account rate for the respective education. 

![Education_Bank_rate](/plots_and_pictures/edu_level_bank_rate.png)

It can be seen that someone with a higher or specialised education is more likely to have a bank account. However, it should also be noted that the number of people who have this education is very small, so that a few people with a bank account have a greater influence on the rate. The same applies to the _Other/Don't_ know statement, which also shows a high rate, although there are only very few participants who have made this statement. 

However, if we now also include _No formal education_, _primary education_ and _secondary education_, there is a tendency for higher levels of education to be more likely to have a bank account. 


## Household Size

The survey also asked about the size of the household in which the interviewee lives. We looked at how high the proportion of bank accounts is for classes of the same household size. This can be seen in the following chart. 

The orange bars represent the number of households of the same size (the x-axis shows the number of people living in the household). The number is shown on the left y-axis. 
The green diamonds represent the proportion of bank accounts within this household class. The corresponding scale is located on the right y-axis.

![Houselhold Size](/plots_and_pictures/household_barplot.png)

## Age of the respondent

For the age of the respondents, we have summarised age groups in a 5-year range. these age groups are plotted on the x-axis and the orange bars represent the number of respondents in the age group (related to the left axis).
For each of these classes, we have calculated the proportion of bank accounts. This proportion is shown with green diamonds (values can be read on the right-hand axis).

![Age](/plots_and_pictures/age_barplot.png)

In the graph, we can see that younger people have fewer bank accounts, but the rate increases with age. From the age group 26-30 (which is also the largest group), the rate remains relatively constant. At an older age, the rate decreases again slightly, but there are also fewer participants. Generally speaking, the age of the respondents has hardly any influence on the bank account rate, except for the very young participants. 