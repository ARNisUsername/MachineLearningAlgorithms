CREATE TABLE Customer (
	[Name] varchar(100),
	[Income] float,
	[Money spent] float
);

INSERT INTO Customer ([Name], Income, [Money Spent]) values ('Jon', 60000, 10.25);
INSERT INTO Customer ([Name], Income, [Money Spent]) values ('Jon', 60000, 10.25);
INSERT INTO Customer ([Name], Income, [Money Spent]) values ('Mary', 40000, 5.50);

SELECT * FROM Customer;
SELECT DISTINCT * FROM Customer;
SELECT COUNT(DISTINCT [Name]) FROM Customer;

INSERT INTO Customer ([Name], Income, [Money Spent]) values ('Jon', 10000, 3.00);

SELECT * FROM Customer WHERE [Name] = 'Jon' AND (Income > 50000 OR Income = 10000);

SELECT * FROM Customer ORDER BY Income ASC;
SELECT * FROM Customer ORDER BY Income DESC;
SELECT * FROM Customer ORDER BY [Name], Income ASC;

SELECT * FROM Customer WHERE [Name] IS NULL;

UPDATE Customer
SET [Name] = 'Jon', [Money spent] -= 0.25 WHERE [Name] like 'Jon%' AND Income < 40000;

DELETE FROM Customer WHERE Income = 10000 AND [Money Spent] < 5.0;

SELECT TOP 2 * FROM Customer WHERE [Name] like 'J%';
SELECT TOP 10 PERCENT * FROM Customer; 

SELECT MIN([Money Spent]) AS MostMoneySpent FROM Customer; 

SELECT COUNT([Money Spent]) AS MoreThan6 FROM Customer WHERE [Money Spent] > 6;
SELECT AVG([Money Spent]) AS AvgMoreThan6 FROM Customer WHERE [Money Spent] > 6;
SELECT SUM([Money Spent]) AS SumMoney FROM Customer;

SELECT * FROM Customer WHERE [Name] NOT LIKE '%ar%';

SELECT * FROM Customer WHERE [Name] NOT IN ('Jonn', 'Mary');

SELECT * FROM Customer WHERE [Money Spent] BETWEEN 5.5 AND 6.5;

USE Customer
SELECT TOP 10 PERCENT * FROM Test;

SELECT school_setting AS Setting,  school_type AS [Type] FROM Test 
WHERE school_setting != 'Urban'; 

CREATE TABLE table1 (
	TableId int,
	Income float,
	Job varchar(50),
);
CREATE TABLE table2 (
	TableId int,
	[Money Spent] float
);

INSERT INTO table1 (TableId, Income, Job) VALUES (3, 50000, 'Architect');
INSERT INTO table1 (TableId, Income, Job) VALUES (4, 60000, 'Senior Architect');
INSERT INTO table1 (TableId, Income, Job) VALUES (6, 70000, 'Pilot');

INSERT INTO table2 (TableId, [Money Spent]) VALUES (3, 100);
INSERT INTO table2 (TableId, [Money Spent]) VALUES (6, 200);

/* INNER JOIN: Select rows that have the same TableId from both tables */
SELECT table2.TableId, table2.[Money Spent], table1.Income, table1.Job FROM table1
INNER JOIN table2 ON table1.TableId = table2.TableId;

/* LEFT JOIN: Select all rows from table1, but add table2 columns to it if they match */
SELECT table2.TableId, table2.[Money Spent], table1.Income, table1.Job FROM table1
LEFT JOIN table2 ON table1.TableId = table2.TableId;

/* RIGHT JOIN: Select all rows from table2, but add table1 columns to it if they match */
SELECT table2.TableId, table2.[Money Spent], table1.Income, table1.Job FROM table1
RIGHT JOIN table2 ON table1.TableId = table2.TableId;

/* FULL JOIN: Return all records from all tables, and add NULL values to missing info */
SELECT table2.TableId, table2.[Money Spent], table1.Income, table1.Job FROM table1
FULL JOIN table2 ON table1.TableId = table2.TableId;

INSERT INTO table1 (TableId, Income, Job) VALUES (9, 80000, 'Pilot');
SELECT * FROM table1;

/* SELF JOIN: Get income information of 2 customers if they have the same job */
SELECT A.Income as Customer1Income, B.Income as Customer2Income, A.Job
FROM table1 A, table1 B 
WHERE A.TableId != B.TableId AND A.Job = B.Job;

/* UNION: Get distinct TableIds from a combination of both tables */
SELECT TableId FROM table1 
UNION 
SELECT TableId from table2;

/* GROUP BY: Count the number of rows with the same Job in table1, and display its count and the job */
SELECT COUNT(TableId), Job FROM table1
GROUP BY Job
HAVING COUNT(TableId) > 1 /* Only display jobs with more than 1 instance */
ORDER BY COUNT(TableId) ASC;

/* ANY: Select the Income off of table1 where the tableId matches the tableId from a subset of table2 */
SELECT Income FROM table1 WHERE TableId = ANY (SELECT TableId FROM table2 WHERE [Money Spent] > 100);

/*InsertId, Income, Job  Money Spent from the inner join into a new table called CombTable */

SELECT table1.TableId, Income, Job, [Money Spent] INTO CombTable
FROM table1 
INNER JOIN table2 ON table1.TableId = table2.TableId;

/* INSERT INTO: Insert Duplicate rows to CombTable where Income > 60000 */

USE Customer;
INSERT INTO CombTable 
SELECT * FROM CombTable
WHERE Income > 60000;

/* CASE: Order by Income if the Income column isn't null, else order by Money Spent */

SELECT Income, Job, [Money Spent] FROM CombTable
ORDER BY 
CASE
	WHEN Income IS NOT NULL THEN Income
	ELSE [Money Spent]
END 
DESC;

/* ALTER TABLE: Add a new column to a table with a specified type */

ALTER TABLE CombTable
ADD NanValue int;

/* ALTER COLUMN: Change the datatype of a column */

ALTER TABLE CombTable
ALTER COLUMN NanValue float; 

/* Add Income to itself multiplied by the column NanValue, but add 0 if NanValue is NAN */
SELECT Job, Income + (Income * ISNULL(NanValue, 0))
FROM CombTable;

/* CREATE PROCEDURE: Create a function given parameters(optional) to recycle code */

CREATE PROCEDURE SelectCustomers (@Income float = 50000) AS
BEGIN
SELECT * FROM CombTable WHERE Income > @Income
END;

EXEC SelectCustomers;
EXEC SelectCustomers @Income = 60000;

/* Primary Key: Constraint to uniquely identify every row */
CREATE TABLE CustomerInfo (
	CustomerID int NOT NULL PRIMARY KEY,
	FirstName varchar(100),
	LastName varchar(150),
	Income float NOT NULL CHECK (Income > -1) /* Check: Ensure data is correct */
);

/* Foreign key: Ties CustomerID from this table to the other, so a customer id cannot be added
unless that id is in the CustomerInfo table */

CREATE TABLE BuyingInfo (
	BuyingID int IDENTITY(1,1) PRIMARY KEY, /* Auto Increment: Automatic Indexing */
	AmountBought float NOT NULL DEFAULT 0.0, /* Add default */
	TotalSpending float NOT NULL DEFAULT 0.0,
	CustomerID int FOREIGN KEY REFERENCES CustomerInfo(CustomerID)
);

INSERT INTO CustomerInfo (CustomerID, FirstName, LastName, Income) VALUES (1, 'Jon', 'Adams', 10000),
(2, 'Amazon', 'Bezos', 50000), (3, 'Epic', 'Human', 120000);

INSERT INTO BuyingInfo (AmountBought, TotalSpending, CustomerID) VALUES (10, 10, 1),
(50, 100, 2), (5, 15, 3);

/* Add a check constraint */
ALTER TABLE BuyingInfo
ADD CONSTRAINT NotNegative CHECK (AmountBought > -1 AND TotalSpending > -1);

/* Drop a check constraint */
ALTER TABLE BuyingInfo
DROP CONSTRAINT NotNegative;

/* Add and remove a default */
ALTER TABLE BuyingInfo
ADD CONSTRAINT Default1 DEFAULT 1.0 FOR AmountBought;

ALTER TABLE BuyingInfo
DROP CONSTRAINT Default1;

/* Create and drop unique index from tables */
CREATE UNIQUE INDEX indexCustomer
ON CustomerInfo (CustomerID);

DROP INDEX CustomerInfo.indexCustomer;
