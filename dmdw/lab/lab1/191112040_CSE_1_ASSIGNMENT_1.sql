
-- Creating a MySQL table according to the given schema. 
-- The fields that were continuos were marked as float, and other 
-- fields with fixed values as VARCHARs.

CREATE TABLE socialinfo (
	id INT NOT NULL auto_increment,
    age float,
    workclass VARCHAR(100),
    fnlwt float,
    education varchar(100),
    educationnum float,
    maritalstatus varchar(100),
    occupation varchar(100),
    relationship varchar(100),
    race varchar(100),
    sex varchar(100),
    capitalgain float,
    capitalloss float,
    hoursperweek float,
    nativecountry varchar(100),
    salary varchar(100),
    primary key (id)
);


-- After creating the table, we load the local infile and store it in our table
load data local infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Sample.txt'
into table socialinfo
fields terminated by ','
lines terminated by '\n'
(age, workclass, fnlwt, education, educationnum, maritalstatus, occupation, relationship,
race, sex, capitalgain, capitalloss, hoursperweek, salary);

-- showing the top 10 results
select * from socialinfo;

-- showing number of entries in the dataset
select count(*) from socialinfo;