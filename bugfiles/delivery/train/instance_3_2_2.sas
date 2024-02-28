begin_version
3
end_version
begin_metric
1
end_metric
4
begin_variable
var0
-1
9
Atom at(t1, c_0_0)
Atom at(t1, c_0_1)
Atom at(t1, c_0_2)
Atom at(t1, c_1_0)
Atom at(t1, c_1_1)
Atom at(t1, c_1_2)
Atom at(t1, c_2_0)
Atom at(t1, c_2_1)
Atom at(t1, c_2_2)
end_variable
begin_variable
var1
-1
2
Atom empty(t1)
<none of those>
end_variable
begin_variable
var2
-1
10
Atom at(p1, c_0_0)
Atom at(p1, c_0_1)
Atom at(p1, c_0_2)
Atom at(p1, c_1_0)
Atom at(p1, c_1_1)
Atom at(p1, c_1_2)
Atom at(p1, c_2_0)
Atom at(p1, c_2_1)
Atom at(p1, c_2_2)
Atom carrying(t1, p1)
end_variable
begin_variable
var3
-1
10
Atom at(p2, c_0_0)
Atom at(p2, c_0_1)
Atom at(p2, c_0_2)
Atom at(p2, c_1_0)
Atom at(p2, c_1_1)
Atom at(p2, c_1_2)
Atom at(p2, c_2_0)
Atom at(p2, c_2_1)
Atom at(p2, c_2_2)
Atom carrying(t1, p2)
end_variable
4
begin_mutex_group
3
2 9
3 9
1 0
end_mutex_group
begin_mutex_group
9
0 0
0 1
0 2
0 3
0 4
0 5
0 6
0 7
0 8
end_mutex_group
begin_mutex_group
10
2 0
2 1
2 2
2 3
2 4
2 5
2 6
2 7
2 8
2 9
end_mutex_group
begin_mutex_group
10
3 0
3 1
3 2
3 3
3 4
3 5
3 6
3 7
3 8
3 9
end_mutex_group
begin_state
8
0
1
0
end_state
begin_goal
2
2 4
3 4
end_goal
60
begin_operator
drop-package t1 p1 c_0_0
1
0 0
2
0 1 -1 0
0 2 9 0
1
end_operator
begin_operator
drop-package t1 p1 c_0_1
1
0 1
2
0 1 -1 0
0 2 9 1
1
end_operator
begin_operator
drop-package t1 p1 c_0_2
1
0 2
2
0 1 -1 0
0 2 9 2
1
end_operator
begin_operator
drop-package t1 p1 c_1_0
1
0 3
2
0 1 -1 0
0 2 9 3
1
end_operator
begin_operator
drop-package t1 p1 c_1_1
1
0 4
2
0 1 -1 0
0 2 9 4
1
end_operator
begin_operator
drop-package t1 p1 c_1_2
1
0 5
2
0 1 -1 0
0 2 9 5
1
end_operator
begin_operator
drop-package t1 p1 c_2_0
1
0 6
2
0 1 -1 0
0 2 9 6
1
end_operator
begin_operator
drop-package t1 p1 c_2_1
1
0 7
2
0 1 -1 0
0 2 9 7
1
end_operator
begin_operator
drop-package t1 p1 c_2_2
1
0 8
2
0 1 -1 0
0 2 9 8
1
end_operator
begin_operator
drop-package t1 p2 c_0_0
1
0 0
2
0 1 -1 0
0 3 9 0
1
end_operator
begin_operator
drop-package t1 p2 c_0_1
1
0 1
2
0 1 -1 0
0 3 9 1
1
end_operator
begin_operator
drop-package t1 p2 c_0_2
1
0 2
2
0 1 -1 0
0 3 9 2
1
end_operator
begin_operator
drop-package t1 p2 c_1_0
1
0 3
2
0 1 -1 0
0 3 9 3
1
end_operator
begin_operator
drop-package t1 p2 c_1_1
1
0 4
2
0 1 -1 0
0 3 9 4
1
end_operator
begin_operator
drop-package t1 p2 c_1_2
1
0 5
2
0 1 -1 0
0 3 9 5
1
end_operator
begin_operator
drop-package t1 p2 c_2_0
1
0 6
2
0 1 -1 0
0 3 9 6
1
end_operator
begin_operator
drop-package t1 p2 c_2_1
1
0 7
2
0 1 -1 0
0 3 9 7
1
end_operator
begin_operator
drop-package t1 p2 c_2_2
1
0 8
2
0 1 -1 0
0 3 9 8
1
end_operator
begin_operator
move t1 c_0_0 c_0_1
0
1
0 0 0 1
1
end_operator
begin_operator
move t1 c_0_0 c_1_0
0
1
0 0 0 3
1
end_operator
begin_operator
move t1 c_0_1 c_0_0
0
1
0 0 1 0
1
end_operator
begin_operator
move t1 c_0_1 c_0_2
0
1
0 0 1 2
1
end_operator
begin_operator
move t1 c_0_1 c_1_1
0
1
0 0 1 4
1
end_operator
begin_operator
move t1 c_0_2 c_0_1
0
1
0 0 2 1
1
end_operator
begin_operator
move t1 c_0_2 c_1_2
0
1
0 0 2 5
1
end_operator
begin_operator
move t1 c_1_0 c_0_0
0
1
0 0 3 0
1
end_operator
begin_operator
move t1 c_1_0 c_1_1
0
1
0 0 3 4
1
end_operator
begin_operator
move t1 c_1_0 c_2_0
0
1
0 0 3 6
1
end_operator
begin_operator
move t1 c_1_1 c_0_1
0
1
0 0 4 1
1
end_operator
begin_operator
move t1 c_1_1 c_1_0
0
1
0 0 4 3
1
end_operator
begin_operator
move t1 c_1_1 c_1_2
0
1
0 0 4 5
1
end_operator
begin_operator
move t1 c_1_1 c_2_1
0
1
0 0 4 7
1
end_operator
begin_operator
move t1 c_1_2 c_0_2
0
1
0 0 5 2
1
end_operator
begin_operator
move t1 c_1_2 c_1_1
0
1
0 0 5 4
1
end_operator
begin_operator
move t1 c_1_2 c_2_2
0
1
0 0 5 8
1
end_operator
begin_operator
move t1 c_2_0 c_1_0
0
1
0 0 6 3
1
end_operator
begin_operator
move t1 c_2_0 c_2_1
0
1
0 0 6 7
1
end_operator
begin_operator
move t1 c_2_1 c_1_1
0
1
0 0 7 4
1
end_operator
begin_operator
move t1 c_2_1 c_2_0
0
1
0 0 7 6
1
end_operator
begin_operator
move t1 c_2_1 c_2_2
0
1
0 0 7 8
1
end_operator
begin_operator
move t1 c_2_2 c_1_2
0
1
0 0 8 5
1
end_operator
begin_operator
move t1 c_2_2 c_2_1
0
1
0 0 8 7
1
end_operator
begin_operator
pick-package t1 p1 c_0_0
1
0 0
2
0 1 0 1
0 2 0 9
1
end_operator
begin_operator
pick-package t1 p1 c_0_1
1
0 1
2
0 1 0 1
0 2 1 9
1
end_operator
begin_operator
pick-package t1 p1 c_0_2
1
0 2
2
0 1 0 1
0 2 2 9
1
end_operator
begin_operator
pick-package t1 p1 c_1_0
1
0 3
2
0 1 0 1
0 2 3 9
1
end_operator
begin_operator
pick-package t1 p1 c_1_1
1
0 4
2
0 1 0 1
0 2 4 9
1
end_operator
begin_operator
pick-package t1 p1 c_1_2
1
0 5
2
0 1 0 1
0 2 5 9
1
end_operator
begin_operator
pick-package t1 p1 c_2_0
1
0 6
2
0 1 0 1
0 2 6 9
1
end_operator
begin_operator
pick-package t1 p1 c_2_1
1
0 7
2
0 1 0 1
0 2 7 9
1
end_operator
begin_operator
pick-package t1 p1 c_2_2
1
0 8
2
0 1 0 1
0 2 8 9
1
end_operator
begin_operator
pick-package t1 p2 c_0_0
1
0 0
2
0 1 0 1
0 3 0 9
1
end_operator
begin_operator
pick-package t1 p2 c_0_1
1
0 1
2
0 1 0 1
0 3 1 9
1
end_operator
begin_operator
pick-package t1 p2 c_0_2
1
0 2
2
0 1 0 1
0 3 2 9
1
end_operator
begin_operator
pick-package t1 p2 c_1_0
1
0 3
2
0 1 0 1
0 3 3 9
1
end_operator
begin_operator
pick-package t1 p2 c_1_1
1
0 4
2
0 1 0 1
0 3 4 9
1
end_operator
begin_operator
pick-package t1 p2 c_1_2
1
0 5
2
0 1 0 1
0 3 5 9
1
end_operator
begin_operator
pick-package t1 p2 c_2_0
1
0 6
2
0 1 0 1
0 3 6 9
1
end_operator
begin_operator
pick-package t1 p2 c_2_1
1
0 7
2
0 1 0 1
0 3 7 9
1
end_operator
begin_operator
pick-package t1 p2 c_2_2
1
0 8
2
0 1 0 1
0 3 8 9
1
end_operator
0
