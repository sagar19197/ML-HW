# Things covered so far -
# 	1. installed pandas,numpy,opencv
#	2. import modules/libraries and using thier functions.
# 	3. Single line comments, docstring
#	4. variable and thier uses. Difference in intialization than cpp or java.
#	5. Strings ,slicing operator, s.strip(),s.upper(),s.lower(),s.replace("1","2")
#		5.1. formating in string :- put {} in strings and then - s.format(variable1)
#		5.2 or s = f"hello {variable1}" - both 5.1 and 5.2 will replace {} or {variable} by the value of variables (similar to printf);
#	6. len(), range(), type(), id()  functions + operators- **,//,%,and,or,not
#	7. for loop, while loop, if statements,break,continue + indentations
#	8. Typecasting by- int(),float(),str()
#	9. Collections in python-
#		9.1 List - [1,2,3](similar to arraylist) - append,insert,pop,remove,del
#		9.2 Tuple - (1,2,3)immutable
#		9.3 Set - {1,2,3} unique elements - add,update,discard,remove,pop-random,intersection,union
#		9.4 Dictionary -{"heelo":2,3:3} - unique key,d["heelo"],d["s"]=1,del d["s]
#	10. Input take - input("enter input");
#	11. Functions - def fun(a,b):


#import cv2;
#import math;


# This is a comment
print("Hello world\n");
if(1<2):
	print("hello");

#print(math.gcd(3,3));

a = 34;
b = "ss";
c = 4.2;

print(a/c);
print(type(a));

# typecastingi
# int(), float(),str()
a = "41";
b = 31;
c = 24.22;
print("string converted to float",float(int(a)));
print("int converted to string",str(float(b)));
print("float coverted to string", str(int(c)));

# Multiline strings
a = """This is multi-line
	String which is 
assigned to a""";
print(a);

# Slicing - 
a = "HelloWorld";
print("a is assinged to",a);
print("slicing applying to 2:5",a[2:5]);
print("len(a) = ",len(a));

a = "     "+a+"      ";
print("a now has leading and trailing spaces",a);
print("Removing those spaces",a.strip());

#Upper(), lower,replace
a = a.strip();
print("lowercase",a.lower());
print("uppercase",a.upper());
print("replacing",a.replace("l","g"));
b = "sgaga, sadds, asdada";
print("replacing",b.replace(", ", "\n"));

# fstring - 
b = "We are running {} program, welcome.";
c = b.format(a);
print(c);
c = f"We are running {a} program";
print(c);

# Extra operators-
a = 3;
print("a^2=",a**2);
print("a//2=",a//2);
print("a%2 =",a%2);

# Collections in python-

# 1. List

lst = [1,3,4,5,6,7];
print("Created list",lst);
#append method-
lst.append([1,2,3]);
print("Appended in the list",lst);
#pop method -
lst.pop();
print("Pop in the list",lst);
#remove method -
lst.remove(4);
print("Remove in the list",lst);
#del operator for deleting the index value
del lst[2];
print("Del operator deleted index2",lst);
# Slicing operator in list - 
print("SLicing operator",lst[0:4:2]);
#insert in list - 
lst.insert(2,10000.23);
print("Inserted in list",lst);


# 2. Tuple - 
tp = (1,2,3);
print(type(tp));
print(tp);
# Not a tuple- 
tp = (1);
print("Value changed - ",type(tp));
tp = (1,2,3);
# Not allowed -
#tp[1] = 100;
#Tuple to list -
lst = list(tp);
print("Converted tuple to list ",lst);
lst[1] = 100.231;
tp = tuple(lst);
print("Converted list to tuple",tp);


# 3. Set -
s = {1,2,3,3,3,2,2,2,4,5,1,1,2};
print(type(s));
print(s);
#add -
s.add(100);
print("Added 100 to set",s);
s.add("helleo");
print("Added string to set",s);
s.update(["ss",12321.3,12,"heel"]);
print("UPdate in set",s);
#s.pop(); It will pop random element
print("Pop in set",s);
s.remove("helleo");
print("Remove in set",s);
s.discard("hellow");
print("Discard in set",s);
s2 = {2,3,4,5,"i am new set"};
print("set 2",s2);
print("Intersection",s.intersection(s2));
print("Union",s.union(s2));


# 4. Dictionary - 
d = {"Name":"Crazy XYZ","Class":12,"Section":"A",0:100};
print(type(d),"\n");
print("We created Dictioary\n",d);
print(d["Name"]);
d["Name"] = "xyz";
print(d);
d["H1"] = "dd";
print(d);
del d["H1"];


# Taking input and elif condition - 
a = int(input("Enter your age :"));
print("Your age is",a);
b = True;
if(a > 18):
	print("You are eligible for driving");
elif(a == 18):
	print("You are turned to 18");
else:
	if(a<10 and not b):
		print("Are you kidding me, how old are you");
	else:
		print("You are not eligible, focus on studies");


#LOOPS-

for i in range(0,11):
	print(i);
print("Printing dictionary");
for i in d:
	print(i);
print("Priniting set");
for i in s:
	print(i);

print("While loop");
i = 0;
while(i<5):
	print(i);
	i+=1;


# Functions-
def sums(a,b):
	print("I am in function");
	return a+b;

print(sums(13,32));

def star(a):
	for i in range(1,a+1):
		print((" "*(a-i)) + (i*"*")+ (i-1)*"*");

a = int(input("Number of stars want:"));
star(a);
