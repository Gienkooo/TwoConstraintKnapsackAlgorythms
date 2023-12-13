int: n;
set of int: ITEMS = 1..n;
array[ITEMS] of int: values; 
array[ITEMS] of int: weights;
array[ITEMS] of int: sizes; 
int: maxweight;
int: maxsize;

array[ITEMS] of var 0..1: x;

constraint sum(i in ITEMS)(x[i] * weights[i]) <= maxweight;
constraint sum(i in ITEMS)(x[i] * sizes[i]) <= maxsize;

var int: totalValue = sum(i in ITEMS)(x[i] * values[i]);

solve maximize totalValue;