int: n;
set of int: ITEMS = 1..n;
array[ITEMS] of int: values; 
array[ITEMS] of int: weights;
array[ITEMS] of int: sizes; 
int: maxW;
int: maxS;

array[ITEMS] of var 0..1: x;

constraint sum(i in ITEMS)(x[i] * weights[i]) <= maxW;
constraint sum(i in ITEMS)(x[i] * sizes[i]) <= maxS;

var int: totalValue = sum(i in ITEMS)(x[i] * values[i]);

solve maximize totalValue;

output [totalValue"];