from munkres import Munkres, print_matrix
import numpy as np
import pandas as pd
import random
import math
import operator
from itertools import combinations
import time
from docplex.mp.model import Model

start_time_code = time.time()

def create_matrix(file):
    open_inst = open(file)
    matrix = list()
    for line in open_inst:
        line=line.rstrip().split()
        for i in range(len(line)):
            line[i] = int(line[i])
        matrix.append(line)
    return matrix

def munkres_method(matrix):
    m = Munkres()
    indexes = m.compute(matrix)
    #print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    values = dict()
    for row, column in indexes:
        value = matrix[row][column]
        values[(row,column)] = value
        total += value

    #print("Values type", type(values))
    values = sorted(values.items(), key=lambda x: x[1], reverse=False)


    #print("values in munkres", values)
    #for i in values:
     #   print(i)
    #print("Finished")
        #print(f'({row}, {column}) -> {value}')
    #print(f'Minimum cost: {total}')
    return total, values

def optimal_sol(matrix_np,k):
    all_solutions = dict()
    matrix = matrix_np.copy()
    n = len(matrix_np)
    n_nodes = list(range(n))
    #print(n_nodes)
    del_all_rows = list(combinations(n_nodes, n - k))
    del_all_cols = list(combinations(n_nodes, n - k))
    #print("dell all rows:", del_all_rows)
    #print("dell all cols:", len(del_all_cols))
    #print("Number of possible solutions: ", len(del_all_cols)**2)
    for i in del_all_rows:
        for j in del_all_cols:
            #print(j, "j element of del all cols ---------")
            matrix_i_j = np.delete(np.delete(matrix, i, 0), j, 1)
            #print(matrix_i_j)
            #print(i,j)
            all_solutions[(i,j)] = munkres_method(matrix_i_j.tolist())[0]

    #sorted_solutions = sorted(all_solutions.items(), key=lambda x: x[1], reverse=True)

    #for i in sorted_solutions:
     #   print(i[0], i[1])

    #for k,v in all_solutions.items():
     #   print(k,v)

    optimal_value = max(all_solutions.values())
    optimal_nodes_delete = max(all_solutions, key = all_solutions.get)
    #print("Optimal solution", optimal_value, "\nRows, cols to delete", optimal_nodes_delete)
    return optimal_value, optimal_nodes_delete

def attempt_0(matrix_np,k):
    # DELETE ROWS

    del_rows = random.sample(range(n),n-k)
    #print("Rows to delete: ", del_rows)

    matrix_np = np.delete(matrix_np, del_rows, 0) #O FOR DELETING ROWS
    #print("After deleting rows:")
    #print(matrix_np)

    # DELETE COLUMNS
    del_cols = random.sample(range(n),n-k)
    #print("Cols to delete: ", del_cols)

    matrix_np = np.delete(matrix_np, del_cols, 1)#1 FOR DELETING COLS
    #print("Reduced matrix:")
    #print(matrix_np)
    current_solution = munkres_method(matrix_np.tolist())[0]
    return current_solution, matrix_np, del_rows, del_cols

def attempt_1(matrix_np,k):
    # DELETE ROWS
    mins_rows = matrix_np.min(axis=1)
    #print("", mins_rows, "mins_rows ")
    del_rows = np.argsort(mins_rows)[:n-k]
    #print("Rows to delete: ", del_rows)

    matrix_np = np.delete(matrix_np, del_rows, 0) #O FOR DELETING ROWS
    #print("After deleting rows:")
    #print(matrix_np)

    # DELETE COLUMNS
    mins_cols = matrix_np.min(axis=0)
    #print("", mins_cols, "mins_cols ")
    del_cols = np.argsort(mins_cols)[:n-k]
    #print("Cols to delete: ", del_cols)

    matrix_np = np.delete(matrix_np, del_cols, 1)#1 FOR DELETING COLS
    #print("Reduced matrix:")
    #print(matrix_np)
    return matrix_np

def attempt_2(matrix_np,k):
    while True:
        min_mean_rows = [round(np.mean(np.sort(i, axis = 0)[:k]),2) for i in matrix_np ]
        min_mean_cols = [round(np.mean(np.sort(i, axis = 0)[:k]),2) for i in np.transpose(matrix_np)]
        min_row = np.min(min_mean_rows)
        min_col = np.min(min_mean_cols)
        #print(min_mean_rows)
        #print("Index min row: ",np.where(min_mean_rows == min_row)[0][0] )
        #print(min_mean_cols)
        #print("Index min col: ", np.where(min_mean_cols == min_col)[0][0])
        rows, cols = matrix_np.shape[0], matrix_np.shape[1]

        if rows ==k and cols == k:
            #print("Final reduced matrix reached: ")
            #print(matrix_np)
            break

        elif cols == k or min_row <= min_col and rows > k:
            #print("A row has the minimum value, delete that row")
            matrix_np = np.delete(matrix_np, np.where(min_mean_rows == min_row)[0][0], 0)
            #print(matrix_np)
        else:
            #print("A col has the minimum value, delete that col")
            matrix_np = np.delete(matrix_np, np.where(min_mean_cols == min_col)[0][0], 1)
            #print(matrix_np)
    return matrix_np

def attempt_3(matrix_np,k):
    while True:
        mean_rows = [round(np.mean(i),2) for i in matrix_np ]
        mean_cols = [round(np.mean(i),2) for i in np.transpose(matrix_np)]
        min_row = np.min(mean_rows)
        min_col = np.min(mean_cols)
        #print(mean_rows)
        #print("Index min row: ",np.where(mean_rows == min_row)[0][0], min_row )
        #print(mean_cols)
        #print("Index min col: ", np.where(mean_cols == min_col)[0][0], min_col)
        rows, cols = matrix_np.shape[0], matrix_np.shape[1]

        if rows ==k and cols == k:
            #print("Final reduced matrix reached: ")
            #print(matrix_np)
            break

        elif cols == k or min_row <= min_col and rows > k:
            #print("A row has the minimum value, delete that row")
            matrix_np = np.delete(matrix_np, np.where(mean_rows == min_row)[0][0], 0)
            #print(matrix_np)
        else:
            #print("A col has the minimum value, delete that col")
            matrix_np = np.delete(matrix_np, np.where(mean_cols == min_col)[0][0], 1)
            #print(matrix_np)
    return matrix_np

def attempt_4(original_matrix_np, k):
    del_rows = list()
    del_cols = list()
    matrix_np = original_matrix_np.copy()
    while True:
        rows, cols = matrix_np.shape[0], matrix_np.shape[1]

        min_mean_rows = [round(np.mean(np.sort(i, axis=0)[:min(n - k, cols)]), 2) for i in matrix_np]
        min_mean_cols = [round(np.mean(np.sort(i, axis=0)[:min(n - k, rows)]), 2) for i in np.transpose(matrix_np)]
        min_row = np.min(min_mean_rows)
        min_col = np.min(min_mean_cols)

        # print(min_mean_rows)
        # print("Index min row: ",np.where(min_mean_rows == min_row)[0])
        # print(min_mean_cols)
        # print("Index min col: ", np.where(min_mean_cols == min_col)[0])

        if rows == k and cols == k:
            # print("Final reduced matrix reached deleting 1 by 1: ")
            # print(matrix_np)
            break

        elif cols == k or min_row <= min_col and rows > k:
            add_rows = 0
            row_to_delete = np.where(min_mean_rows == min_row)[0][0]
            # print("A row has the minimum value, delete row", row_to_delete)
            matrix_np = np.delete(matrix_np, np.where(min_mean_rows == min_row)[0][0], 0)
            # add_rows = len([i for i in del_rows if i<=row_to_delete])

            while add_rows < len([i for i in del_rows if i <= row_to_delete + add_rows]):
                add_rows = len([i for i in del_rows if i <= row_to_delete + add_rows])

            # while row_to_delete + add_rows in del_rows:
            #   add_rows = len([i for i in del_rows if i<=row_to_delete + add_rows])

            # print("Append row: ", row_to_delete + add_rows)
            # print(matrix_np)
            del_rows.append(row_to_delete + add_rows)

        else:
            add_cols = 0
            col_to_delete = np.where(min_mean_cols == min_col)[0][0]
            # print("A col has the minimum value, delete the col", col_to_delete)
            matrix_np = np.delete(matrix_np, np.where(min_mean_cols == min_col)[0][0], 1)

            while add_cols < len([i for i in del_cols if i <= col_to_delete + add_cols]):
                add_cols = len([i for i in del_cols if i <= col_to_delete + add_cols])

            # print("Append col: ", col_to_delete + add_cols)
            # print(matrix_np)
            del_cols.append(col_to_delete + add_cols)

        # print(del_rows)
        # print(del_cols)
    # reduced_1 = np.delete(np.delete(original_matrix_np, del_rows, 0), del_cols, 1)
    # print("Final reduced matrix reached deleting all at once: ")
    # print(reduced_1)
    # if (reduced_1 == matrix_np).all():
    #  print("EQUAL")
    # else: print("--------------------- DIFFERENT ----------------------")
    current_solution = munkres_method(matrix_np.tolist())[0]

    return current_solution, matrix_np, sorted(del_rows), sorted(del_cols)

def attempt_5(original_matrix_np,k):

    del_rows = list()
    del_cols = list()
    matrix_np = original_matrix_np.copy()
    while True:
        rows, cols = matrix_np.shape[0], matrix_np.shape[1]
        min_mean_rows = [round(np.mean(np.sort(i, axis = 0)[:max(2,math.floor(cols/2))]),2) for i in matrix_np]
        min_mean_cols = [round(np.mean(np.sort(i, axis = 0)[:max(2,math.floor(rows/2))]),2) for i in np.transpose(matrix_np)]
        min_row = np.min(min_mean_rows)
        min_col = np.min(min_mean_cols)
        #print(min_mean_rows)
        #print("Index min row: ",np.where(min_mean_rows == min_row)[0])
        #print(min_mean_cols)
        #print("Index min col: ", np.where(min_mean_cols == min_col)[0])


        if rows ==k and cols == k:
            #print("Final reduced matrix reached deleting 1 by 1: ")
            #print(matrix_np)
            break

        elif cols == k or min_row <= min_col and rows > k:
            add_rows = 0
            row_to_delete = np.where(min_mean_rows == min_row)[0][0]
            #print("A row has the minimum value, delete row", row_to_delete)
            matrix_np = np.delete(matrix_np, np.where(min_mean_rows == min_row)[0][0], 0)
            #add_rows = len([i for i in del_rows if i<=row_to_delete])

            while add_rows < len([i for i in del_rows if i <= row_to_delete + add_rows]):
                add_rows = len([i for i in del_rows if i <= row_to_delete + add_rows])

            #while row_to_delete + add_rows in del_rows:
             #   add_rows = len([i for i in del_rows if i<=row_to_delete + add_rows])

            #print("Append row: ", row_to_delete + add_rows)
            #print(matrix_np)
            del_rows.append(row_to_delete + add_rows)

        else:
            add_cols = 0
            col_to_delete = np.where(min_mean_cols == min_col)[0][0]
            #print("A col has the minimum value, delete the col", col_to_delete)
            matrix_np = np.delete(matrix_np, np.where(min_mean_cols == min_col)[0][0], 1)

            while add_cols < len([i for i in del_cols if i <= col_to_delete + add_cols]):
                add_cols = len([i for i in del_cols if i <= col_to_delete + add_cols])

            #print("Append col: ", col_to_delete + add_cols)
            #print(matrix_np)
            del_cols.append(col_to_delete + add_cols)

        #print(del_rows)
        #print(del_cols)
    #reduced_1 = np.delete(np.delete(original_matrix_np, del_rows, 0), del_cols, 1)
    #print("Final reduced matrix reached deleting all at once: ")
    #print(reduced_1)
    #if (reduced_1 == matrix_np).all():
      #  print("EQUAL")
    #else: print("--------------------- DIFFERENT ----------------------")
    current_solution = munkres_method(matrix_np.tolist())[0]

    return current_solution, matrix_np, sorted(del_rows), sorted(del_cols)

def attempt_499(matrix_np,k):
    while True:
        min_mean_rows = [round(np.mean(np.sort(i, axis = 0)[:min(n-k,k)]),2) for i in matrix_np ]
        min_mean_cols = [round(np.mean(np.sort(i, axis = 0)[:min(n-k,k)]),2) for i in np.transpose(matrix_np)]
        min_row = np.min(min_mean_rows)
        min_col = np.min(min_mean_cols)
        #print(min_mean_rows)
        #print("Index min row: ",np.where(min_mean_rows == min_row)[0] )
        #print(min_mean_cols)
        #print("Index min col: ", np.where(min_mean_cols == min_col)[0])
        rows, cols = matrix_np.shape[0], matrix_np.shape[1]

        if rows ==k and cols == k:
            #print("Final reduced matrix reached: ")
            #print(matrix_np)
            break

        elif cols == k or min_row <= min_col and rows > k:
            #print("A row has the minimum value, delete that row")
            matrix_np = np.delete(matrix_np, np.where(min_mean_rows == min_row)[0][0], 0)
            #print(matrix_np)
        else:
            #print("A col has the minimum value, delete that col")
            matrix_np = np.delete(matrix_np, np.where(min_mean_cols == min_col)[0][0], 1)
            #print(matrix_np)
    return matrix_np

#-----------------------------------------------------
#N_1_1: verified, it is ok. Mixed strategy (1 row, 1 col) DONT USE IT
def neighborhood_1_1(current_solution,original_matrix, del_rows, del_cols, prev, munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Non deleted rows: ", non_del_rows)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    #first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    #current_solution = munkres_method(first_matrix.tolist())[0]
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    for old in range(n-k):
        new_del_rows_0 = np.delete(del_rows, [old])
        #print("Delete element ", del_rows[old])
        for new in range(k):
            new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
            #print("New del rows: ", new_del_rows)
            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
            solution = munkres_method(matrix.tolist())[0]
            munkres += 1

            if solution > current_solution:
                row_neighbors[tuple(new_del_rows)] = solution
                #print(matrix.tolist())

        if len(row_neighbors)>0:
            break
        #print("Row neighb")
        #for i,j in row_neighbors.items(): print(i,j)

        #print("Non deleted cols: ", non_del_cols)
        #print("Deleted cols ", del_cols)

    for old in range(n-k):
        new_del_cols_0 = np.delete(del_cols, [old])
        #print("Delete element ", del_cols[old])
        for new in range(k):
            new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
            #print("New del cols: ", new_del_cols)
            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
            solution = munkres_method(matrix.tolist())[0]
            munkres += 1

            if solution > current_solution:
                col_neighbors[tuple(new_del_cols)] = solution
                #print(matrix.tolist())
        if len(col_neighbors)>0:
            break

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())


    if max_value_row == 0 and max_value_col == 0:
        print("Local optimum reached with this neighborhood ", current_solution)
        print("Total munkres", munkres)
        improvement = 0

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev, munkres

#N_3_2 Verified. OK. Half rows, half cols, half rows, half cols DONT USE IT
def neighborhood_3_2(current_solution,original_matrix, del_rows, del_cols, prev, munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Non deleted rows: ", non_del_rows)
    #print("Non deleted cols: ", non_del_cols)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]
    order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)
    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)


    if prev == "col":
        #print("Rows 1st, Rows 1")
        for new in order_rows[:int(k/2)]:
            new_del_rows0 = np.append(del_rows, non_del_rows[new])
            #print("New del rows0", new_del_rows0)
            for old in range(n-k):
                new_del_rows = np.delete(new_del_rows0, [old])
                #print("New del rows", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                munkres+=1

                if solution > current_solution:
                    row_neighbors[tuple(new_del_rows)] = solution
                    #print(matrix.tolist())

            if len(row_neighbors)>0:
                print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                prev = "row"
                break

        #print("Row neighb")
        #for i,j in row_neighbors.items(): print(i,j)

        #print("Non deleted cols: ", non_del_cols)
        #print("Deleted cols ", del_cols)
        if len(row_neighbors) == 0:
            #print("Rows 1st, cols 1")
            for new in order_cols[:int(k / 2)]:
                new_del_cols0 = np.append(del_cols, non_del_cols[new])
                #print("New del col0", new_del_cols0)
                for old in range(n - k):
                    new_del_cols = np.delete(new_del_cols0, [old])
                    #print("New del cols", new_del_cols)
                    #print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres+=1

                    if solution > current_solution:
                        col_neighbors[tuple(new_del_cols)] = solution
                        #print(matrix.tolist())
                if len(col_neighbors)>0:
                    print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                    prev = "col"
                    break

            if len(col_neighbors)==0:
                #print("Rows 1st, rows 2")
                for new in order_rows[int(k / 2):]:
                    new_del_rows0 = np.append(del_rows, non_del_rows[new])
                    # print("New del rows0", new_del_rows0)
                    for old in range(n - k):
                        new_del_rows = np.delete(new_del_rows0, [old])
                        # print("New del rows", new_del_rows)
                        matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                        solution = munkres_method(matrix.tolist())[0]
                        munkres += 1

                        if solution > current_solution:
                            row_neighbors[tuple(new_del_rows)] = solution
                            # print(matrix.tolist())

                    if len(row_neighbors) > 0:
                        print("Improvement found deleting row", new_del_rows[len(new_del_rows) - 1])
                        prev = "row"
                        break

                if len(row_neighbors) == 0:
                    #print("Rows 1st, cols 2")
                    for new in order_cols[int(k / 2):]:
                        new_del_cols0 = np.append(del_cols, non_del_cols[new])
                        # print("New del col0", new_del_cols0)
                        for old in range(n - k):
                            new_del_cols = np.delete(new_del_cols0, [old])
                            # print("New del cols", new_del_cols)
                            # print("New del cols: ", new_del_cols)
                            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                            solution = munkres_method(matrix.tolist())[0]
                            munkres += 1

                            if solution > current_solution:
                                col_neighbors[tuple(new_del_cols)] = solution
                                # print(matrix.tolist())
                        if len(col_neighbors) > 0:
                            print("Improvement found deleting col", new_del_cols[len(new_del_cols) - 1])
                            prev = "col"
                            break

    else:
        #print("Cols 1st col 1")
        for new in order_cols[:int(k / 2)]:
            new_del_cols0 = np.append(del_cols, non_del_cols[new])
            #print("New del col0", new_del_cols0)
            for old in range(n - k):
                new_del_cols = np.delete(new_del_cols0, [old])
                #print("New del cols", new_del_cols)
                matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                munkres += 1

                if solution > current_solution:
                    col_neighbors[tuple(new_del_cols)] = solution
                    # print(matrix.tolist())

            if len(col_neighbors) > 0:
                print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                prev = "col"
                break

            # print("Row neighb")
            # for i,j in row_neighbors.items(): print(i,j)

            # print("Non deleted cols: ", non_del_cols)
            # print("Deleted cols ", del_cols)
        if len(col_neighbors) == 0:
            #print("Cols 1st rows 1")

            for new in order_rows[:int(k / 2)]:
                new_del_rows0 = np.append(del_rows, non_del_rows[new])
                #print("New del rows0", new_del_rows0)
                for old in range(n - k):
                    new_del_rows = np.delete(new_del_rows0, [old])
                    #print("New del rows", new_del_rows)
                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres += 1

                    if solution > current_solution:
                        row_neighbors[tuple(new_del_rows)] = solution
                        # print(matrix.tolist())
                if len(row_neighbors) > 0:
                    print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                    prev = "row"
                    break

            if len(row_neighbors)==0:
                #print("Cols 1st col 2")

                for new in order_cols[int(k / 2):]:
                    new_del_cols0 = np.append(del_cols, non_del_cols[new])
                    # print("New del col0", new_del_cols0)
                    for old in range(n - k):
                        new_del_cols = np.delete(new_del_cols0, [old])
                        # print("New del cols", new_del_cols)
                        matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                        solution = munkres_method(matrix.tolist())[0]
                        munkres += 1

                        if solution > current_solution:
                            col_neighbors[tuple(new_del_cols)] = solution
                            # print(matrix.tolist())

                    if len(col_neighbors) > 0:
                        print("Improvement found deleting col", new_del_cols[len(new_del_cols) - 1])
                        prev = "col"
                        break

                if len(col_neighbors) == 0:
                    #print("Cols 1st rows 2")

                    for new in order_rows[int(k / 2):]:
                        new_del_rows0 = np.append(del_rows, non_del_rows[new])
                        # print("New del rows0", new_del_rows0)
                        for old in range(n - k):
                            new_del_rows = np.delete(new_del_rows0, [old])
                            # print("New del rows", new_del_rows)
                            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                            solution = munkres_method(matrix.tolist())[0]
                            munkres += 1

                            if solution > current_solution:
                                row_neighbors[tuple(new_del_rows)] = solution
                                # print(matrix.tolist())
                        if len(row_neighbors) > 0:
                            print("Improvement found deleting row", new_del_rows[len(new_del_rows) - 1])
                            prev = "row"
                            break

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())

    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0
        print("Total munkres", munkres)

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev, munkres

#I dont know what is this:
def neighborhood_39(original_matrix, del_rows, del_cols, prev):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Non deleted rows: ", non_del_rows)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]
    order_rows = [i[0][0] for i in values]
    print("order_rows",order_rows)
    order_cols = [i[0][1] for i in values]
    print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    if prev == "col":

        for old in order_rows:
            new_del_rows_0 = np.delete(del_rows, [old])
            #print("Delete element ", del_rows[old])
            for new in range(k):
                new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                #print("New del rows: ", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                if solution > current_solution:
                    row_neighbors[tuple(new_del_rows)] = solution
                    #print(matrix.tolist())

            if len(row_neighbors)>0:
                print("Improvement found on row", old)
                prev = "row"
                break

        #print("Row neighb")
        #for i,j in row_neighbors.items(): print(i,j)

        #print("Non deleted cols: ", non_del_cols)
        #print("Deleted cols ", del_cols)
        if len(row_neighbors) == 0:

            for old in order_cols:
                new_del_cols_0 = np.delete(del_cols, [old])
                #print("Delete element ", del_cols[old])
                for new in range(k):
                    new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                    #print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]

                    if solution > current_solution:
                        col_neighbors[tuple(new_del_cols)] = solution
                        #print(matrix.tolist())
                if len(col_neighbors)>0:
                    print("Improvement found on col", old)
                    prev = "col"
                    break

    else:
        for old in order_cols:
            new_del_cols_0 = np.delete(del_cols, [old])
            # print("Delete element ", del_rows[old])
            for new in range(k):
                new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                # print("New del rows: ", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                if solution > current_solution:
                    col_neighbors[tuple(new_del_cols)] = solution
                    # print(matrix.tolist())

            if len(col_neighbors) > 0:
                print("Improvement found on col", old)
                prev = "col"
                break

            # print("Row neighb")
            # for i,j in row_neighbors.items(): print(i,j)

            # print("Non deleted cols: ", non_del_cols)
            # print("Deleted cols ", del_cols)
        if len(col_neighbors) == 0:

            for old in order_rows:
                new_del_rows_0 = np.delete(del_rows, [old])
                # print("Delete element ", del_cols[old])
                for new in range(k):
                    new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                    # print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]

                    if solution > current_solution:
                        row_neighbors[tuple(new_del_rows)] = solution
                        # print(matrix.tolist())
                if len(row_neighbors) > 0:
                    print("Improvement found on row", old)
                    prev = "row"
                    break

    #print("Col neighb")
    #for i,j in col_neighbors.items(): print(i,j)

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())
    #else: #print("Non better row swap")
    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())
    #else: #print("Non better col swap")

    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        #print("Swap the best row\nNew del_rows:")
        #print(del_rows, max_value_row)
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        #print("Swap the best col\nNew del_cols:")
        #print(del_cols, max_value_col)
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev

#I dont know what is this:
def neighborhood_49(original_matrix, del_rows, del_cols, prev):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows = np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Non deleted rows: ", non_del_rows)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]
    order_rows = [i[0][0] for i in values]
    #_rows",order_rows)
    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    if prev == "col":
        print("Rows 1st, Rows 1")
        for old in order_rows[:int(k/2)]:
            new_del_rows_0 = np.delete(del_rows, [old])
            #print("Delete element ", del_rows[old])
            for new in range(k):
                new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                #print("New del rows: ", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                if solution > current_solution:
                    row_neighbors[tuple(new_del_rows)] = solution
                    #print(matrix.tolist())

            if len(row_neighbors)>0:

                print("Improvement found on row", old)
                prev = "row"
                break

        #print("Row neighb")
        #for i,j in row_neighbors.items(): print(i,j)

        #print("Non deleted cols: ", non_del_cols)
        #print("Deleted cols ", del_cols)
        if len(row_neighbors) == 0:
            print("Rows 1st, Cols 1")
            for old in order_cols[:int(k/2)]:
                new_del_cols_0 = np.delete(del_cols, [old])
                #print("Delete element ", del_cols[old])
                for new in range(k):
                    new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                    #print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]

                    if solution > current_solution:
                        col_neighbors[tuple(new_del_cols)] = solution
                        #print(matrix.tolist())
                if len(col_neighbors)>0:

                    print("Improvement found on col", old)
                    prev = "col"
                    break

            if len(col_neighbors) == 0:
                print("Rows 1st, Rows 2")
                for old in order_rows[int(k / 2):]:
                    new_del_rows_0 = np.delete(del_rows, [old])
                    # print("Delete element ", del_rows[old])
                    for new in range(k):
                        new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                        # print("New del rows: ", new_del_rows)
                        matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                        solution = munkres_method(matrix.tolist())[0]
                        if solution > current_solution:
                            row_neighbors[tuple(new_del_rows)] = solution
                            # print(matrix.tolist())

                    if len(row_neighbors) > 0:

                        print("Improvement found on row", old)
                        prev = "row"
                        break

                if len(row_neighbors) == 0 :
                    print("Rows 1st, Cols 2")
                    for old in order_cols[int(k / 2):]:
                        new_del_cols_0 = np.delete(del_cols, [old])
                        # print("Delete element ", del_cols[old])
                        for new in range(k):
                            new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                            # print("New del cols: ", new_del_cols)
                            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                            solution = munkres_method(matrix.tolist())[0]

                            if solution > current_solution:
                                col_neighbors[tuple(new_del_cols)] = solution
                                # print(matrix.tolist())
                        if len(col_neighbors) > 0:

                            print("Improvement found on col", old)
                            prev = "col"
                            break



    else:
        print("Cols 1st, Cols 1")
        for old in order_cols[:int(k/2)]:
            new_del_cols_0 = np.delete(del_cols, [old])
            # print("Delete element ", del_rows[old])
            for new in range(k):
                new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                # print("New del rows: ", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                if solution > current_solution:
                    col_neighbors[tuple(new_del_cols)] = solution
                    # print(matrix.tolist())

            if len(col_neighbors) > 0:

                print("Improvement found on col", old)
                prev = "col"
                break

            # print("Row neighb")
            # for i,j in row_neighbors.items(): print(i,j)

            # print("Non deleted cols: ", non_del_cols)
            # print("Deleted cols ", del_cols)
        if len(col_neighbors) == 0:
            print("Cols 1st, Rows 1")
            for old in order_rows[:int(k/2)]:
                new_del_rows_0 = np.delete(del_rows, [old])
                # print("Delete element ", del_cols[old])
                for new in range(k):
                    new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                    # print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]

                    if solution > current_solution:
                        row_neighbors[tuple(new_del_rows)] = solution
                        # print(matrix.tolist())
                if len(row_neighbors) > 0:

                    print("Improvement found on row", old)
                    prev = "row"
                    break

            if len(row_neighbors) == 0 :
                print("Cols 1st, Cols 2")
                for old in order_cols[int(k/2):]:
                    new_del_cols_0 = np.delete(del_cols, [old])
                    # print("Delete element ", del_rows[old])
                    for new in range(k):
                        new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                        # print("New del rows: ", new_del_rows)
                        matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                        solution = munkres_method(matrix.tolist())[0]
                        if solution > current_solution:
                            col_neighbors[tuple(new_del_cols)] = solution
                            # print(matrix.tolist())

                    if len(col_neighbors) > 0:

                        print("Improvement found on col", old)
                        prev = "col"
                        break

                    # print("Row neighb")
                    # for i,j in row_neighbors.items(): print(i,j)

                    # print("Non deleted cols: ", non_del_cols)
                    # print("Deleted cols ", del_cols)
                if len(col_neighbors) == 0:
                    print("Cols 1st, Rows 2")
                    for old in order_rows[int(k/2):]:
                        new_del_rows_0 = np.delete(del_rows, [old])
                        # print("Delete element ", del_cols[old])
                        for new in range(k):
                            new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                            # print("New del cols: ", new_del_cols)
                            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                            solution = munkres_method(matrix.tolist())[0]

                            if solution > current_solution:
                                row_neighbors[tuple(new_del_rows)] = solution
                                # print(matrix.tolist())
                        if len(row_neighbors) > 0:

                            print("Improvement found on row", old)
                            prev = "row"
                            break

    #print("Col neighb")
    #for i,j in col_neighbors.items(): print(i,j)

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())
    #else: #print("Non better row swap")
    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())
    #else: #print("Non better col swap")

    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        #print("Swap the best row\nNew del_rows:")
        #print(del_rows, max_value_row)
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        #print("Swap the best col\nNew del_cols:")
        #print(del_cols, max_value_col)
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev
#-----------------------------------------------------

#N1: verified, it is ok. Best improvement strategy
def neighborhood_1(current_solution,original_matrix, del_rows, del_cols, prev, munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Non deleted rows: ", non_del_rows)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    #first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    #current_solution = munkres_method(first_matrix.tolist())[0]
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    for old in range(n-k):
        new_del_rows_0 = np.delete(del_rows, [old])
        #print("Delete element ", del_rows[old])
        for new in range(k):
            new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
            #print("New del rows: ", new_del_rows)
            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
            solution = munkres_method(matrix.tolist())[0]
            munkres += 1

            if solution > current_solution:
                row_neighbors[tuple(new_del_rows)] = solution
                #print(matrix.tolist())

        #if len(row_neighbors)>0:
         #   break
        #print("Row neighb")
        #for i,j in row_neighbors.items(): print(i,j)

        #print("Non deleted cols: ", non_del_cols)
        #print("Deleted cols ", del_cols)

    for old in range(n-k):
        new_del_cols_0 = np.delete(del_cols, [old])
        #print("Delete element ", del_cols[old])
        for new in range(k):
            new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
            #print("New del cols: ", new_del_cols)
            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
            solution = munkres_method(matrix.tolist())[0]
            munkres += 1

            if solution > current_solution:
                col_neighbors[tuple(new_del_cols)] = solution
                #print(matrix.tolist())
        #if len(col_neighbors)>0:
         #   break
    #print("Col neighb")
    #for i,j in col_neighbors.items(): print(i,j)

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())
    #else: #print("Non better row swap")
    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())
    #else: #print("Non better col swap")

    if max_value_row == 0 and max_value_col == 0:
        print("Local optimum reached with this neighborhood ", current_solution)
        #print("Total munkres", munkres)
        improvement = 0

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        #print("Swap the best row\nNew del_rows:")
        #print(del_rows, max_value_row)
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        #print("Swap the best col\nNew del_cols:")
        #print(del_cols, max_value_col)
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev, munkres

#N2: verified, it is ok. First improvement strategy (1 row, 1 col)
def neighborhood_2(current_solution, original_matrix, del_rows, del_cols, prev, munkres):
    # print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows = np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    # print("Non deleted rows: ", non_del_rows)
    # print("Deleted rows ", del_rows)
    # print("Number of neigbors to visit: ", 2*(n-k)*k)

    # first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    # current_solution = munkres_method(first_matrix.tolist())[0]
    # del_rows =list(range(n-k))
    # print("Del rows,", del_rows)

    for old in range(n - k):
        new_del_rows_0 = np.delete(del_rows, [old])
        # print("Delete element ", del_rows[old])
        for new in range(k):
            new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
            # print("New del rows: ", new_del_rows)
            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
            solution = munkres_method(matrix.tolist())[0]
            munkres += 1

            if solution > current_solution:
                row_neighbors[tuple(new_del_rows)] = solution
                break

        if len(row_neighbors) > 0:
            break
        # print("Row neighb")
        # for i,j in row_neighbors.items(): print(i,j)

        # print("Non deleted cols: ", non_del_cols)
        # print("Deleted cols ", del_cols)

    for old in range(n - k):
        new_del_cols_0 = np.delete(del_cols, [old])
        # print("Delete element ", del_cols[old])
        for new in range(k):
            new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
            # print("New del cols: ", new_del_cols)
            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
            solution = munkres_method(matrix.tolist())[0]
            munkres += 1

            if solution > current_solution:
                col_neighbors[tuple(new_del_cols)] = solution
                break

        if len(col_neighbors) > 0:
            break

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors) > 0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors) > 0:
        max_value_col = max(col_neighbors.values())

    if max_value_row == 0 and max_value_col == 0:
        print("Local optimum reached with this neighborhood ", current_solution)
        print("Total munkres", munkres)
        improvement = 0

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key=row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev, munkres

#N3 Verified. OK Optimized first improvement strategy
def neighborhood_3(current_solution,original_matrix, del_rows, del_cols, prev, munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Non deleted rows: ", non_del_rows)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    #current_solution, values = munkres_method(first_matrix.tolist())[:]
    #order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)
    #order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    if prev == "col":

        for old in range(n-k):
            new_del_rows_0 = np.delete(del_rows, [old])
            #print("Delete element ", del_rows[old])
            for new in range(k):
                new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                #print("New del rows: ", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                munkres += 1

                if solution > current_solution:
                    row_neighbors[tuple(new_del_rows)] = solution
                    #print(matrix.tolist())

            if len(row_neighbors)>0:
                #print("Improvement found on row", old)
                prev = "row"
                break

        #print("Row neighb")
        #for i,j in row_neighbors.items(): print(i,j)

        #print("Non deleted cols: ", non_del_cols)
        #print("Deleted cols ", del_cols)
        if len(row_neighbors) == 0:

            for old in range(n-k):
                new_del_cols_0 = np.delete(del_cols, [old])
                #print("Delete element ", del_cols[old])
                for new in range(k):
                    new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                    #print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres += 1

                    if solution > current_solution:
                        col_neighbors[tuple(new_del_cols)] = solution
                        #print(matrix.tolist())
                if len(col_neighbors)>0:
                    #print("Improvement found on col", old)
                    prev = "col"
                    break

    else:
        for old in range(n-k):
            new_del_cols_0 = np.delete(del_cols, [old])
            # print("Delete element ", del_rows[old])
            for new in range(k):
                new_del_cols = np.append(new_del_cols_0, non_del_cols[new])
                # print("New del rows: ", new_del_rows)
                matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                munkres += 1

                if solution > current_solution:
                    col_neighbors[tuple(new_del_cols)] = solution
                    # print(matrix.tolist())

            if len(col_neighbors) > 0:
                #print("Improvement found on col", old)
                prev = "col"
                break

            # print("Row neighb")
            # for i,j in row_neighbors.items(): print(i,j)

            # print("Non deleted cols: ", non_del_cols)
            # print("Deleted cols ", del_cols)
        if len(col_neighbors) == 0:

            for old in range(n-k):
                new_del_rows_0 = np.delete(del_rows, [old])
                # print("Delete element ", del_cols[old])
                for new in range(k):
                    new_del_rows = np.append(new_del_rows_0, non_del_rows[new])
                    # print("New del cols: ", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres += 1

                    if solution > current_solution:
                        row_neighbors[tuple(new_del_rows)] = solution
                        # print(matrix.tolist())
                if len(row_neighbors) > 0:
                    #print("Improvement found on row", old)
                    prev = "row"
                    break

    #print("Col neighb")
    #for i,j in col_neighbors.items(): print(i,j)

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())
    #else: #print("Non better row swap")
    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())
    #else: #print("Non better col swap")

    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0
        print("Total munkres", munkres)

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        #print("Swap the best row\nNew del_rows:")
        #print(del_rows, max_value_row)
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        #print("Swap the best col\nNew del_cols:")
        #print(del_cols, max_value_col)
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev, munkres

#N4 Verified. OK First all rows, then all colums
def neighborhood_4(current_solution,original_matrix, del_rows, del_cols, prev, munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)

    #print("Non deleted rows: ", non_del_rows)
    #print("Non deleted cols: ", non_del_cols)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]
    order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)
    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)

    if prev == "col":

        for new in order_rows:
            new_del_rows0 = np.append(del_rows, non_del_rows[new])

            for old in range(n-k):
                new_del_rows = np.delete(new_del_rows0, [old])
                matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                munkres+=1

                if solution > current_solution:
                    row_neighbors[tuple(new_del_rows)] = solution

            if len(row_neighbors)>0:
                #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                prev = "row"
                break

        if len(row_neighbors) == 0:

            for new in order_cols:
                new_del_cols0 = np.append(del_cols, non_del_cols[new])

                for old in range(n - k):
                    new_del_cols = np.delete(new_del_cols0, [old])
                    matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres+=1

                    if solution > current_solution:
                        col_neighbors[tuple(new_del_cols)] = solution

                if len(col_neighbors)>0:
                    #print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                    prev = "col"
                    break

    else:
        for new in order_cols:
            new_del_cols0 = np.append(del_cols, non_del_cols[new])

            for old in range(n - k):
                new_del_cols = np.delete(new_del_cols0, [old])
                matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                solution = munkres_method(matrix.tolist())[0]
                munkres += 1

                if solution > current_solution:
                    col_neighbors[tuple(new_del_cols)] = solution

            if len(col_neighbors) > 0:
                #print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                prev = "col"
                break

        if len(col_neighbors) == 0:

            for new in order_rows:
                new_del_rows0 = np.append(del_rows, non_del_rows[new])

                for old in range(n - k):
                    new_del_rows = np.delete(new_del_rows0, [old])
                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres += 1

                    if solution > current_solution:
                        row_neighbors[tuple(new_del_rows)] = solution

                if len(row_neighbors) > 0:
                    #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                    prev = "row"
                    break

    max_value_row, max_value_col = 0, 0
    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())


    if max_value_row == 0 and max_value_col == 0:
        improvement = 0
        print("Total munkres", munkres)

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    return current_solution, del_rows, del_cols, improvement, prev, munkres

#N5 IS BELOW

#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#

def calculate_k_inst(instances, attempt, local_search, size, swap, matheuristic):
    min_costs, munkres_list, local_search_count_list, diff_sols_list =list(), list(), list(), list()
    math_improvement_list = list()
    math_results_list = list()
    math_counter_list = list()
    math_fp_impr_global = list()

    selected_oi_dj = list()

    avoided_munkres_list = list()

    #print("For ks ", ks)

    for instance in instances:

        min_costs_inst, munkres_inst, local_search_count_inst, diff_sols_inst = list(), list(), list(), list()
        math_improvement_inst = list()
        math_results_inst = list()
        math_counter_inst =  list()
        math_fp_impr_inst = list()

        selected_oi_dj_inst = list()

        avoided_munkres_inst = list()

        file = False
        if size == 10:
            file = "Inst_10x10_" + str(instance) + ".txt"
        elif size == 30:
            file = "Inst_30x30_" + str(instance) + ".txt"
        elif size == 50:
            file = "Inst_50x50_" + str(instance) + ".txt"
        elif size == 75:
            file = "Inst_75x75_" + str(instance) + ".txt"
        elif size == 100:
            file = "Inst_100x100_" + str(instance) + ".txt"

        if file == False:

            original_matrix = np.array([[5,4,5,4],
                           [5,3,8,15],
                           [20,9,7,6],
                           [1,2,10,12]])

        else:
            original_matrix = np.array(create_matrix(file))
            #print("Rows, columns of original matrix")
            #print(len(original_matrix), len(np.transpose(original_matrix)))
        #print("Original matrix:\n", original_matrix)

        for k in ks:
            current_solution = 0
            print("k = ",k)
            if attempt == 10:
                current_solution = optimal_sol(original_matrix, k)[0]
            else:
                if attempt == 0:
                    start_time = time.time()
                    a0_results = attempt_0(original_matrix, k)
                    current_solution, reduced_matrix, del_rows, del_cols = a0_results[:]
                elif attempt == 1:
                    reduced_matrix = attempt_1(original_matrix,k)
                elif attempt == 2:
                    reduced_matrix = attempt_2(original_matrix, k)
                elif attempt == 3:
                    reduced_matrix = attempt_3(original_matrix, k)
                elif attempt == 4:
                    a4_results = attempt_4(original_matrix, k)
                    current_solution, reduced_matrix, del_rows, del_cols = a4_results[:]

                elif attempt == 5:
                    start_time = time.time()
                    a5_results = attempt_5(original_matrix, k)
                    current_solution, reduced_matrix, del_rows, del_cols = a5_results[:]


                elif attempt == 45:
                    start_time = time.time()
                    a5_results = attempt_5(original_matrix, k)
                    current_solution, reduced_matrix, del_rows, del_cols = a5_results[:]
                    a4_results = attempt_4(original_matrix, k)
                    current_solution4, reduced_matrix4, del_rows4, del_cols4 = a4_results[:]
                    if current_solution4 > current_solution:
                        current_solution, reduced_matrix, del_rows, del_cols = a4_results[:]

            #Calculate the first solution
                if current_solution == 0 :
                    current_solution = munkres_method(reduced_matrix.tolist())[0]
                else:
                    munkres = 1

                print("Attempt solution", current_solution)

                solution_dict = dict()
                solution_references_list = list()
                repeated_sol = 0
                local_search_count = 0
                repeated_sol_temp = 0
                #num_swaps_rows = min(math.ceil(k / 7), k, size - k)
                #num_swaps_cols = min(math.ceil(k / 4), k, size - k)

                # num_swaps_rows = 1
                # num_swaps_cols = 1

                #Swap (2).
                #num_swaps_rows = max(1,math.floor(min(k, size - k)/6))
                #num_swaps_cols = max(1,math.floor(min(k, size - k)/6))

                #Swap (2)
                num_swaps_rows = max(1, math.ceil(min(k, size - k) / 4))
                num_swaps_cols = max(1, math.ceil(min(k, size - k) / 4))

                #if k/n >=10.7:
                 #   print("True, add more distance to swap")
                  #  num_swaps_rows += int(10 * (n - k) / n)
                   # num_swaps_cols += int(10 * (n - k) / n)

                # print("number of swaps:", num_swaps)
                current_best = 0
                del_rows_best = list()
                del_cols_best = list()

                references = np.append(sorted(del_rows), sorted(del_cols)).tolist()
                del_rows_best, del_cols_best = del_rows, del_cols

                no_impr = 1
                inst_k_sol__delrows_delcols = dict()
                global_munkres = 0
                global_avoided_munkres = 0

                while local_search == True:

                    total_time = time.time() - start_time
                    if total_time >= time_limit:
                        print("Time limit reached. Time used: ", round(total_time,2))
                        break

                    current_solution, munkres, del_rows, del_cols, avoided_munkres = apply_neighb(original_matrix,
                                                                                 del_rows, del_cols, start_time)[:]

                    global_munkres+=munkres
                    global_avoided_munkres+=avoided_munkres

                    if current_solution> current_best:

                        print("Local Optimum Found")
                        print("Time:", time.time()-start_time )
                        print("Improvement found, change bests rows and cols, reset number of swaps")

                        current_best, del_rows_best, del_cols_best = current_solution, del_rows, del_cols

                        current_best_matrix = np.delete(np.delete(original_matrix, del_rows_best, 0), del_cols_best, 1).tolist()
                        values = munkres_method(current_best_matrix)[1]

                        #for i in values:
                         #   print(i)

                        order_rows = [i[0][0] for i in values]
                        #print("order_rows",order_rows)
                        order_cols = [i[0][1] for i in values]
                        #print("order_cols", order_cols)

                        no_impr = 1
                        repeated_sol_temp = 0
                        num_swaps_rows = max(1, math.ceil(min(k, size - k) / 4))
                        num_swaps_cols = max(1, math.ceil(min(k, size - k) / 4))

                        #if k / n >= 10.7:
                            #num_swaps_rows += int(10*(n - k) / n)
                            #num_swaps_cols += int(10*(n - k) / n)

                    else:
                        no_impr+=1

                    local_search_count+=1

                    del_rows_del_cols = np.append(sorted(del_rows), sorted(del_cols)).tolist()

                    #print("LOCAL SEARCH FINISHED--------------")
                    #print("del_rows_del_cols", del_rows_del_cols)
                    #print("solution_references_list", solution_references_list)


                    if del_rows_del_cols not in solution_references_list:
                        solution_dict[tuple(del_rows_del_cols)] = current_solution
                        solution_references_list.append(del_rows_del_cols)
                        print("New solution")
                        print("-----------------------------------------------------")
                        repeated_now = False


                    else:
                        print("Solution already found!!")
                        print("-----------------------------------------------------")
                        repeated_sol+=1
                        repeated_sol_temp+=1
                        repeated_now = True



                    if swap == True:
                        adaptive = True
                        print("No improvments in a row ", no_impr)
                        #if repeated_sol_temp >0 and repeated_now == True:
                         #   if (repeated_sol_temp + 5) % 10 ==0 :
                          #      num_swaps_rows = min(k, size - k, num_swaps_rows+1)

                          #  if repeated_sol_temp % 10 ==0:
                           #     num_swaps_cols = min(k, size - k, num_swaps_cols + 1)
                        if adaptive == True:
                            if (no_impr + 5) % 10 ==0:
                                num_swaps_rows = min(num_swaps_rows+1, k, size -k)
                            if no_impr  % 10 == 0 or (repeated_now == True and repeated_sol_temp ==2):
                                num_swaps_cols = min(num_swaps_cols+1, k, size -k)

                        print("Total repeated solutions: ", repeated_sol)
                        print("Rows to swap ", num_swaps_rows)
                        print("Cols to swap ", num_swaps_cols)

                        del_rows, del_cols = random_swap(original_matrix, del_rows_best, del_cols_best,
                                                         num_swaps_rows, num_swaps_cols)


                    else:
                        break

                #print("Solutions dictionary\n", solution_dict)

                print("WE FOUND ", repeated_sol, "TIMES AN ALREADY FOUND SOLUTION")


                if local_search == True:
                    best_local_found, references = current_best, [del_rows_best, del_cols_best]
                else:
                    best_local_found = current_solution

                total_solutions = len(solution_dict)

                if swap == False:
                    print("Local search finished")

                else:
                    print("Iterated local search finished")
                    print("TOTAL LOCAL SEARCH DONE: ", local_search_count)
                    print("TOTAL DIFFERENT SOLUTIONS: ", total_solutions)

                print("Best found for [instance, k] after ILS/LS ", instance, k, "= ", best_local_found)
                #print("Reference [deleted rows, deleted cols]= ", references)
                print("********************************** LS/ILS Finished **********************************")
                inst_k_sol__delrows_delcols[instance,k,best_local_found]= del_rows_best,del_cols_best

                if matheuristic:
                    math_improvement = 0
                    math_counter = 0
                    time_matheur = time.time()
                    math_no_impr_counter = 0
                    fixed_percent = fixed_percent_global
                    math_fp_impr = dict()

                    while True:
                        time_used_math = time.time() - time_matheur
                        if time_used_math >= time_limit:
                            print("Matheuristic reached the time limit with", math_improvement, "improvements and", math_counter, "iterations")
                            break
                        math_counter += 1
                        print("Apply Matheuristic, It. ", math_counter)
                        print("Consecutive no improvements: ", math_no_impr_counter)
                        print("Fixed percent of the current best solution: ", fixed_percent)

                        if adaptive_fixed_percent:
                            if math_no_impr_counter != 0 and math_no_impr_counter % 5 == 0:
                                fixed_percent= round(fixed_percent - 0.05, 2)
                                #math_no_impr_counter = 0
                                print("Five iterations without improvements...")
                                print("New fixed percent ", fixed_percent)

                        #THIS WORKS, TRY TOMORROW AFTER TRYING WITHOUT IT
                        #1 if math_no_impr_counter %2 >0: #<-100:#
                            #2 print("Should apply the new algorithm that selects the non fixed to fix...")
                            #variables_matheuristic = fixed_variables(del_rows_best, del_cols_best,
                                                                     #3 fixed_percent, variables_matheuristic, False)[:]
                            #change_afixed_row = False

                        #elif math_no_impr_counter % 14 == 0 and math_no_impr_counter!= 0:
                         #   print("Should change the solution by fixing a deleted row...")
                          #  variables_matheuristic = fixed_variables(del_rows_best, del_cols_best,
                           #                                          fixed_percent, None, False, True,original_matrix)[:]
                           # change_afixed_row = True

                        #4 else:
                        variables_matheuristic = fixed_variables(del_rows_best, del_cols_best, fixed_percent)[:]
                            #change_afixed_row = False

                        rows_to_fix, cols_to_fix, rows_to_delete, cols_to_delete = variables_matheuristic
                        M = 100
                        remaining_time = time_limit-time_used_math



                        y = create_and_solve_model(n, k, M, original_matrix, greater_0, remaining_time, rows_to_fix, cols_to_fix,
                                                   rows_to_delete, cols_to_delete,
                                                   output)
                        # y = [sol_value, time, nb_its, status, rel_gap, o_values, d_values]

                        matheur_best = y[0]

                        current_o_values, current_d_values = y[5], y[6]

                        if greater_0 == True:

                            print("Result obtained by the relaxed AP model: ", matheur_best)

                            range_n = [i for i in range(n)]
                            del_rows_math = [i for i in range(n) if i not in current_o_values]
                            del_cols_math = [i for i in range(n) if i not in current_d_values]

                            selected_matrix = np.delete(np.delete(original_matrix, del_rows_math, 0), del_cols_math, 1).tolist()

                            # print("selected_matrix")
                            # print(selected_matrix)

                            real_result = munkres_method(selected_matrix)[0]

                            if real_result == matheur_best:
                                print("The real solution value is the same")

                            else:
                                print("Result obtained by the real AP model: ", real_result)
                                matheur_best = real_result


                        if matheur_best > current_best:
                            print("Matheuristic improves the current best solution")
                            print("Current fixed percent: ", fixed_percent)
                            print("Current best: ", current_best)
                            print("After matheuristic: ", matheur_best)
                            current_best = matheur_best
                            del_rows_best = del_rows_math
                            del_cols_best = del_cols_math
                            math_improvement += 1
                            math_no_impr_counter=0

                            math_fp_impr[fixed_percent]=math_fp_impr.get(fixed_percent,0) + 1
                            fixed_percent = fixed_percent_global




                        else:
                            print("Matheuristic does not improve current best solution")
                            math_no_impr_counter+=1


                        print("-----------------------------------------------------")


            #print("Selected iodj inst: ")
            #print(selected_oi_dj_inst)

            selected_oi = [i for i in range(n) if i not in del_rows_best]
            selected_dj = [i for i in range(n) if i not in del_cols_best]

            selected_variables = [selected_oi, selected_dj]
            selected_oi_dj_inst.append(selected_variables)

            min_costs_inst.append(best_local_found)

            #munkres_inst.append(global_munkres)
            munkres_inst.append(global_munkres)
            avoided_munkres_inst.append(global_avoided_munkres)

            local_search_count_inst.append(local_search_count)

            diff_sols_inst.append(total_solutions)

            if matheuristic:
                math_improvement_inst.append(math_improvement)

                math_results_inst.append(current_best)

                math_counter_inst.append(math_counter)

                math_fp_impr_inst.append(math_fp_impr)


        print("Instance number ", str(instance))
        print(min_costs_inst)
        min_costs.append(min_costs_inst)
        munkres_list.append(munkres_inst)
        avoided_munkres_list.append(avoided_munkres_inst)

        local_search_count_list.append(local_search_count_inst)
        diff_sols_list.append(diff_sols_inst)
        math_improvement_list.append(math_improvement_inst)
        math_results_list.append(math_results_inst)
        math_counter_list.append(math_counter_inst)
        math_fp_impr_global.append(math_fp_impr_inst)

        selected_oi_dj.append(selected_oi_dj_inst)

    print("For all instances and k wanted results after LS/ILS are ")
    print(np.array(min_costs))
    print("Munkres")
    print(np.array(munkres_list))
    print("Avoided munkres")
    print(np.array(avoided_munkres_list))

    total_avoided_munkres = np.sum(np.array(avoided_munkres_list))
    total_munkres = np.sum(np.array(munkres_list))

    if total_avoided_munkres + total_munkres > 0:
        percent_avoided_munkres = total_avoided_munkres/(total_avoided_munkres+total_munkres)
        print("% of avoided munkres:", percent_avoided_munkres)

    print("Number of local searches:\n", np.array(local_search_count_list))
    print("Number of different solutions found:\n", np.array(diff_sols_list))

    print("Improvements of matheuristic: ")
    print(np.array(math_improvement_list))

    print("Matheuristic results: ")
    print(np.array(math_results_list))

    print("Matheuristic counter: ")
    print(np.array(math_counter_list))

    print("Matheuristic, for what distance were found how many improvements, for each k and instance: ")
    print(np.array(math_fp_impr_global))

    print("Selected oi djs")
    for i in selected_oi_dj:
        print(i)

    #print("Improvements reached by deleting a fixed row: ")
    #print(math_improvement_changing_fixed)

    return min_costs, munkres_list, sorted(del_rows_best), sorted(del_cols_best), \
           original_matrix, local_search_count_list, diff_sols_list, inst_k_sol__delrows_delcols, \
           math_improvement_list, math_results_list, math_counter_list, math_fp_impr_global,\
           selected_oi_dj, avoided_munkres_list

#Neigborhood 3_3: OK. 1 row, 1 col, 1 row, 1 col, etc.
def neighborhood_5(current_solution,original_matrix, del_rows, del_cols, prev, munkres, avoided_munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Selected rows: ", non_del_rows)
    #print("Selected cols: ", non_del_cols)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]

    order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)

    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    #New part of the code
    #print("List to sort")
    #print(values)

    values_rows_dict = dict()
    values_cols_dict = dict()

    for i in values:
        values_rows_dict[i[0][0]] = i[1]
        values_cols_dict[i[0][1]] = i[1]

    values_rows, values_cols = list(), list()

    for i in range(len(values)):
        values_rows.append(values_rows_dict[i])
        values_cols.append(values_cols_dict[i])

    #print("Final sorted vales rows:")
    #print(values_rows)
    #print("Final sorted vales cols:")
    #print(values_cols)


    if prev == "col":
        #print("Rows 1st")

        for i in range(k):

            #print("i = ", i)
            index_row_out = order_rows[i] #False index... delete rows in order of increasing values (max:k)

            row_out = non_del_rows[index_row_out]
            value_out = values_rows[index_row_out]

            new_del_rows0 = np.append(del_rows, row_out)
            #print("Follower's value out", value_out)
            #esto saco
            #print("Del_rows ", del_rows)
            #print("Selected rows ", non_del_rows)
            #print("Row out ", row_out)

            #print("new_del_rows0 ", new_del_rows0)


            #print("New del rows0", new_del_rows0)
            #print("Rows")

            for old in range(n-k):

                new_del_rows = np.delete(new_del_rows0, [old])
                row_in = np.delete(original_matrix[del_rows[old]], del_cols)
                value_in = row_in[values[i][0][1]]
                #print("Follower's value in", value_in)

                # esto saco
                #print("Row in ", del_rows[old])
                #print("New del rows", new_del_rows)
                #print("New selected rows", [i for i in range(n) if i not in new_del_rows])


                #print("Follower's row in",original_matrix[del_rows[old]], row_in)
                #print("values [i]][0][1]",values[i][0][1])


                if value_in > value_out:
                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)

                    solution = munkres_method(matrix.tolist())[0]
                    munkres+=1
                    #print("munkres", munkres)

                    if solution > current_solution:
                        row_neighbors[tuple(new_del_rows)] = solution
                        #print(matrix.tolist())

                else:
                    avoided_munkres += 1
                    #print("HM is not applied, cyl <= axl")


            if len(row_neighbors)>0:
                #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                prev = "row"
                break

            #print("Row neighb")
            #for i,j in row_neighbors.items(): print(i,j)

            #print("Non deleted cols: ", non_del_cols)
            #print("Deleted cols ", del_cols)
            else:
                #print("Cols")
                #print("Rows 1st, cols 1")
                index_col_out = order_cols[i]  # False index... delete rows in order of increasing values (max:k)

                col_out = non_del_cols[index_col_out]
                value_out = values_cols[index_col_out]
                print("checking of value out: ", value_out)
                new_del_cols0 = np.append(del_cols, col_out)
                #print("New del col0", new_del_cols0)
                for old in range(n - k):
                    new_del_cols = np.delete(new_del_cols0, [old])
                    #print("New del cols", new_del_cols)
                    #print("New del cols: ", new_del_cols)
                    #print("ver esto")
                    col_in = np.delete(np.transpose(original_matrix)[del_cols[old]], del_rows)
                    print("Follower's col in", np.transpose(original_matrix)[del_cols[old]], col_in)
                    #print("values [i][0][1]", values[i][0][0])
                    value_in = col_in[values[i][0][0]]
                    print("Follower's value in", value_in)

                    if value_in > value_out:
                        matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                        solution = munkres_method(matrix.tolist())[0]
                        munkres+=1
                        #print("munkres", munkres)

                        if solution > current_solution:
                            col_neighbors[tuple(new_del_cols)] = solution
                            print("Improvement found deleting col", index_col_out, "inserting", col_in, del_cols[old])
                            #print(matrix.tolist())
                    else:
                        avoided_munkres+=1
                        print("HM is not applied, cyl <= axl")

                if len(col_neighbors)>0:
                    print("Improvement found deleting col", index_col_out, "inserting", col_in, del_cols[old])
                    prev = "col"
                    break

            if len(col_neighbors) > 0:
                break

    else:
        #print("Cols 1st")
        for i in range(k):
            #print("i = ", i)
            #print("Cols")
            #new_del_cols0 = np.append(del_cols, non_del_cols[order_cols[i]])
            #print("New del col0", new_del_cols0)
            index_col_out = order_cols[i]  # False index... delete rows in order of increasing values (max:k
            col_out = non_del_cols[index_col_out]
            value_out = values_cols[index_col_out]
            #print("Value out: ", value_out)
            new_del_cols0 = np.append(del_cols, col_out)

            for old in range(n - k):

                new_del_cols = np.delete(new_del_cols0, [old])
                # print("New del cols", new_del_cols)
                # print("New del cols: ", new_del_cols)

                col_in = np.delete(np.transpose(original_matrix)[del_cols[old]], del_rows)
                #print("Follower's col in", np.transpose(original_matrix)[del_cols[old]], col_in)
                #print("values [i][0][1]", values[i][0][0])
                value_in = col_in[values[i][0][0]]
                #print("Follower's value in", value_in)

                if value_in > value_out:
                    #print("New del cols", new_del_cols)
                    matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                    solution = munkres_method(matrix.tolist())[0]
                    munkres += 1
                    #print("munkres", munkres)

                    if solution > current_solution:
                        col_neighbors[tuple(new_del_cols)] = solution
                        # print(matrix.tolist())

                else:
                    avoided_munkres += 1
                    #print("HM is not applied, cyl <= axl")

            if len(col_neighbors) > 0:
                #print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                prev = "col"
                break

                # print("Row neighb")
                # for i,j in row_neighbors.items(): print(i,j)

                # print("Non deleted cols: ", non_del_cols)
                # print("Deleted cols ", del_cols)
            else:
                #print("Rows")

                index_row_out = order_rows[i]  # False index... delete rows in order of increasing values (max:k)

                row_out = non_del_rows[index_row_out]
                value_out = values_rows[index_row_out]

                new_del_rows0 = np.append(del_rows, row_out)

                #print("Del_rows ", del_rows)
                #print("Selected rows ", non_del_rows)
                #print("Row out ", row_out)
                #print("Follower's value out", value_out)
                # print("new_del_rows0 ", new_del_rows0)


                #print("New del rows0", new_del_rows0)
                for old in range(n - k):

                    new_del_rows = np.delete(new_del_rows0, [old])
                    #print("Row in ", del_rows[old])
                    #print("New del rows", new_del_rows)
                    #print("New selected rows", [i for i in range(n) if i not in new_del_rows])

                    row_in = np.delete(original_matrix[del_rows[old]], del_cols)
                    #print("Follower's row in", original_matrix[del_rows[old]], row_in)
                    #print("values [i]][0][1]", values[i][0][1])
                    value_in = row_in[values[i][0][1]]
                    #print("Follower's value in", value_in)

                    if value_in > value_out:
                        matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                        solution = munkres_method(matrix.tolist())[0]
                        munkres += 1
                        #print("munkres", munkres)

                        if solution > current_solution:
                            row_neighbors[tuple(new_del_rows)] = solution
                            # print(matrix.tolist())
                    else:
                        avoided_munkres+=1
                        #print("HM is not applied, cyl <= axl")

                if len(row_neighbors) > 0:
                    #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                    prev = "row"
                    break

            if len(row_neighbors)>0:
                break

    max_value_row, max_value_col = 0, 0

    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())


    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0
        #print("Total munkres", munkres)

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)

    #print("Original matrix:\n", original_matrix)
    #print("Deleted rows: ", del_rows)
    #print("Deleted columns: ", del_cols)
    #print("Reduced matrix:\n", matrix)
    #print("Values of the solution for AP: ", values)

    return current_solution, del_rows, del_cols, improvement, prev, munkres, avoided_munkres


#Same to neigborhood 3_3 but with the new truncation
def neighborhood_5_truncated(current_solution,original_matrix, del_rows, del_cols, prev, munkres, avoided_munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Selected rows: ", non_del_rows)
    #print("Selected cols: ", non_del_cols)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]

    #print("Original followers matrix")
    #print(np.array(first_matrix))

    order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)

    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    #New part of the code
    #print("List to sort")
    #print(values)

    values_rows_dict = dict()
    values_cols_dict = dict()
    coord_row_to_col = dict()
    coord_col_to_row = dict()

    for i in values:
        values_rows_dict[i[0][0]] = i[1]
        values_cols_dict[i[0][1]] = i[1]
        coord_row_to_col[i[0][0]] = i[0][1]
        coord_col_to_row[i[0][1]] = i[0][0]

    #print("coords row to col")
    #print(coord_row_to_col)

    values_rows, values_cols = list(), list()
    rows_order_of_bcols, cols_order_of_brows = list(),list()

    for i in range(len(values)):
        values_rows.append(values_rows_dict[i])
        values_cols.append(values_cols_dict[i])
        rows_order_of_bcols.append(coord_col_to_row[i])
        cols_order_of_brows.append(coord_row_to_col[i])

    #print("Final sorted vales rows:")
    #print(values_rows)
    #print("Final sorted vales cols:")
    #print(values_cols)
    #print("Final coordinates rows of b cols:")
    #print(rows_order_of_bcols)
    #print("Final coordinates cols of b rows:")
    #print(cols_order_of_brows)

    if prev == "col":
        #print("Rows 1st")

        for i in range(k):

            #print("i = ", i)
            index_row_out = order_rows[i] #False index... delete rows in order of increasing values (max:k)
            #print("index_row_out",index_row_out)
            row_out_real_index = non_del_rows[index_row_out] #Row x = [axI, ... axm... axk]



            value_out = values_rows[index_row_out]

            new_del_rows0 = np.append(del_rows, row_out_real_index)

            row_out = first_matrix[index_row_out]

            #print("Follower's value out", value_out)
            #esto saco
            #print("Del_rows ", del_rows)
            #print("Selected rows ", non_del_rows)
            #print("Row out ", row_out, row_out_real_index)

            #print("new_del_rows0 ", new_del_rows0)


            #print("New del rows0", new_del_rows0)
            #print("Rows")

            m = values[i][0][1]
            #print("m = ", m)

            for old in range(n-k):  #Old = index row in
                print("Row in", del_rows[old], "Row out", index_row_out)
                new_del_rows = np.delete(new_del_rows0, [old])
                row_in = np.delete(original_matrix[del_rows[old]], del_cols) #Row y = [CyI, ... Cym... Cyk]

                value_in = row_in[m]
                #print("Follower's value in", value_in)

                # esto saco
                #print("Row in ", row_in, del_rows[old])
                #print("New del rows", new_del_rows)
                #print("New selected rows", [i for i in range(n) if i not in new_del_rows])


                #print("Follower's row in",original_matrix[del_rows[old]], row_in)
                #print("values [i]][0][1]",values[i][0][1])
                do_munkres = True

                if value_in > value_out:

                    for q in range(k):

                        p = rows_order_of_bcols[q]
                        row_p = first_matrix[p]


                        b_q = values_cols[q]
                        b_m = values_rows[index_row_out] #bx of rows = axm = bm of cols
                        old_values_b = b_q + b_m
                        print("b_q + b_m ", b_q, "+", b_m, "=", old_values_b)

                        b_q_new = row_in[q]
                        b_m_new = row_p[m]
                        new_values_b = b_q_new + b_m_new
                        print("b_q' and b_m'",b_q_new,"+", b_m_new, "=", new_values_b)

                        if new_values_b <= old_values_b:
                            print("Truncated neighbor linear!!")
                            avoided_munkres+=1
                            do_munkres = False
                            break

                    if do_munkres:
                        matrix=np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                        print("Doing munkres")
                        solution = munkres_method(matrix.tolist())[0]
                        munkres+=1

                        #print("munkres", munkres)

                        if solution > current_solution:
                            print("Improvement")
                            row_neighbors[tuple(new_del_rows)] = solution
                            #print(matrix.tolist())

                else:
                    avoided_munkres += 1
                    #print("Truncated neighbor constant!!")
                    #print("HM is not applied, cyl <= axl")


            if len(row_neighbors)>0:
                print("Improvement found deleting row", index_row_out)
                print("Inserting row (approximate) ", del_rows[old])
                prev = "row"
                break



            else:
                #print("Cols")
                #print("Rows 1st, cols 1")
                index_col_out = order_cols[i]  # False index... delete rows in order of increasing values (max:k)

                col_out_real_index = non_del_cols[index_col_out]
                value_out = values_cols[index_col_out]
                #print("checking of value out: ", value_out)
                new_del_cols0 = np.append(del_cols, col_out_real_index)

                col_out = np.transpose(first_matrix)[index_col_out]
                #print("col_out ",col_out)

                l = values[i][0][0]
                #print("l = ", l)
                #print("New del col0", new_del_cols0)
                for old in range(n - k):
                    new_del_cols = np.delete(new_del_cols0, [old])
                    #print("New del cols", new_del_cols)
                    #print("New del cols: ", new_del_cols)
                    #print("ver esto")
                    col_in = np.delete(np.transpose(original_matrix)[del_cols[old]], del_rows)
                    #print("Follower's col in", np.transpose(original_matrix)[del_cols[old]], col_in)
                    #print("values [i][0][1]", values[i][0][0])
                    value_in = col_in[l]
                    #print("Follower's value in", value_in)
                    #print("col_in",col_in)
                    do_munkres = True

                    if value_in > value_out:

                        for p in range(k):

                            q = cols_order_of_brows[p]
                            col_q = np.transpose(first_matrix)[q]

                            b_p = values_rows[p]
                            b_l = values_cols[index_col_out]  # bx of rows = axm = bm of cols
                            old_values_b = b_p + b_l
                            #print("b_p + b_m ", b_p, "+", b_l, "=", old_values_b)

                            b_p_new = col_in[p]
                            b_l_new = col_q[l]
                            new_values_b = b_p_new + b_l_new
                            #print("b_p' and b_m'", b_p_new, "+", b_l_new, "=", new_values_b)

                            if new_values_b <= old_values_b:
                                #print("Truncated neighbor linear!!")
                                avoided_munkres += 1
                                do_munkres = False
                                break

                        if do_munkres:
                            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                            #print("Doing munkres")
                            solution = munkres_method(matrix.tolist())[0]
                            munkres += 1

                            # print("munkres", munkres)


                            if solution > current_solution:
                                col_neighbors[tuple(new_del_cols)] = solution
                                #print("Improvement")
                                #print(matrix.tolist())
                    else:
                        avoided_munkres+=1
                        #print("Truncated neighbor constant!!")
                        #print("HM is not applied, cyl <= axl")

                if len(col_neighbors)>0:
                    #print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                    prev = "col"
                    break

            if len(col_neighbors) > 0:
                break

    else:
        #print("Cols 1st")
        for i in range(k):
            #print("i = ", i)
            #print("Cols")
            #new_del_cols0 = np.append(del_cols, non_del_cols[order_cols[i]])
            #print("New del col0", new_del_cols0)
            index_col_out = order_cols[i]  # False index... delete rows in order of increasing values (max:k)

            col_out_real_index = non_del_cols[index_col_out]
            value_out = values_cols[index_col_out]
            # print("checking of value out: ", value_out)
            new_del_cols0 = np.append(del_cols, col_out_real_index)

            col_out = np.transpose(first_matrix)[index_col_out]
            # print("col_out ",col_out)

            l = values[i][0][0]
            #print("l = ", l)
            # print("New del col0", new_del_cols0)
            for old in range(n - k):
                new_del_cols = np.delete(new_del_cols0, [old])
                # print("New del cols", new_del_cols)
                # print("New del cols: ", new_del_cols)
                # print("ver esto")
                col_in = np.delete(np.transpose(original_matrix)[del_cols[old]], del_rows)
                # print("Follower's col in", np.transpose(original_matrix)[del_cols[old]], col_in)
                # print("values [i][0][1]", values[i][0][0])
                value_in = col_in[l]
                #print("Follower's value in", value_in)
                #print("col_in", col_in)
                do_munkres = True

                if value_in > value_out:

                    for p in range(k):

                        q = cols_order_of_brows[p]
                        col_q = np.transpose(first_matrix)[q]

                        b_p = values_rows[p]
                        b_l = values_cols[index_col_out]  # bx of rows = axm = bm of cols
                        old_values_b = b_p + b_l
                        # print("b_p + b_m ", b_p, "+", b_l, "=", old_values_b)

                        b_p_new = col_in[p]
                        b_l_new = col_q[l]
                        new_values_b = b_p_new + b_l_new
                        # print("b_p' and b_m'", b_p_new, "+", b_l_new, "=", new_values_b)

                        if new_values_b <= old_values_b:
                            #print("Truncated neighbor!!")
                            avoided_munkres += 1
                            do_munkres = False
                            break

                    if do_munkres:
                        matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                        # print("Doing munkres")
                        solution = munkres_method(matrix.tolist())[0]
                        munkres += 1

                        # print("munkres", munkres)

                        if solution > current_solution:
                            col_neighbors[tuple(new_del_cols)] = solution
                            # print("Improvement")
                            # print(matrix.tolist())
                else:
                    avoided_munkres += 1
                    #print("Truncated neighbor!!")
                    # print("HM is not applied, cyl <= axl")

            if len(col_neighbors) > 0:
                # print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                prev = "col"
                break

                # print("Row neighb")
                # for i,j in row_neighbors.items(): print(i,j)

                # print("Non deleted cols: ", non_del_cols)
                # print("Deleted cols ", del_cols)

            else:
                #print("Rows now")

                index_row_out = order_rows[i]  # False index... delete rows in order of increasing values (max:k)

                row_out_real_index = non_del_rows[index_row_out]
                value_out = values_rows[index_row_out]

                row_out = first_matrix[index_row_out]

                new_del_rows0 = np.append(del_rows, row_out_real_index)

                #print("Del_rows ", del_rows)
                #print("Selected rows ", non_del_rows)
                #print("Row out ", row_out)
                #print("Follower's value out", value_out)
                # print("new_del_rows0 ", new_del_rows0)
                m = values[i][0][1]
                #print("m = ", m)

                #print("New del rows0", new_del_rows0)

                for old in range(n - k):  # Old = index row in

                    new_del_rows = np.delete(new_del_rows0, [old])
                    row_in = np.delete(original_matrix[del_rows[old]], del_cols)  # Row y = [CyI, ... Cym... Cyk]

                    value_in = row_in[m]
                    # print("Follower's value in", value_in)

                    # esto saco
                    #print("Row in ", row_in, del_rows[old])
                    # print("New del rows", new_del_rows)
                    # print("New selected rows", [i for i in range(n) if i not in new_del_rows])

                    # print("Follower's row in",original_matrix[del_rows[old]], row_in)
                    # print("values [i]][0][1]",values[i][0][1])
                    do_munkres = True

                    if value_in > value_out:

                        for q in range(k):

                            p = rows_order_of_bcols[q]
                            row_p = first_matrix[p]

                            b_q = values_cols[q]
                            b_m = values_rows[index_row_out]  # bx of rows = axm = bm of cols
                            old_values_b = b_q + b_m
                            #print("b_q + b_m ", b_q, "+", b_m, "=", old_values_b)

                            b_q_new = row_in[q]
                            b_m_new = row_p[m]
                            new_values_b = b_q_new + b_m_new
                            #print("b_q' and b_m'", b_q_new, "+", b_m_new, "=", new_values_b)

                            if new_values_b <= old_values_b:
                                #print("Truncated neighbor!!")
                                avoided_munkres += 1
                                do_munkres = False
                                break

                        if do_munkres:
                            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                            #print("Doing munkres")
                            solution = munkres_method(matrix.tolist())[0]
                            munkres += 1

                            # print("munkres", munkres)

                            if solution > current_solution:
                                row_neighbors[tuple(new_del_rows)] = solution
                                # print(matrix.tolist())

                    else:
                        avoided_munkres += 1
                        # print("HM is not applied, cyl <= axl")

                if len(row_neighbors) > 0:
                    #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                    prev = "row"
                    break

            if len(row_neighbors)>0:
                break

    max_value_row, max_value_col = 0, 0

    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())


    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0
        #print("Total munkres", munkres)

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)

    #print("Original matrix:\n", original_matrix)
    #print("Deleted rows: ", del_rows)
    #print("Deleted columns: ", del_cols)
    #print("Reduced matrix:\n", matrix)
    #print("Values of the solution for AP: ", values)

    return current_solution, del_rows, del_cols, improvement, prev, munkres, avoided_munkres

#Same to neigborhood 3_3 but with the final truncation
def neighborhood_5_final(current_solution,original_matrix, del_rows, del_cols, prev, munkres, avoided_munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    row_neighbors = dict()
    col_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Selected rows: ", non_del_rows)
    #print("Selected cols: ", non_del_cols)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]

    #print("Original followers matrix")
    #print(np.array(first_matrix))

    order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)

    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    #New part of the code
    #print("List to sort")
    #print(values)

    values_rows_dict = dict()
    values_cols_dict = dict()
    coord_row_to_col = dict()
    coord_col_to_row = dict()

    for i in values:
        values_rows_dict[i[0][0]] = i[1]
        values_cols_dict[i[0][1]] = i[1]
        coord_row_to_col[i[0][0]] = i[0][1]
        coord_col_to_row[i[0][1]] = i[0][0]

    #print("coords row to col")
    #print(coord_row_to_col)

    values_rows, values_cols = list(), list()
    rows_order_of_bcols, cols_order_of_brows = list(),list()

    for i in range(len(values)):
        values_rows.append(values_rows_dict[i])
        values_cols.append(values_cols_dict[i])
        rows_order_of_bcols.append(coord_col_to_row[i])
        cols_order_of_brows.append(coord_row_to_col[i])

    #print("Final sorted vales rows:")
    #print(values_rows)
    #print("Final sorted vales cols:")
    #print(values_cols)
    #print("Final coordinates rows of b cols:")
    #print(rows_order_of_bcols)
    #print("Final coordinates cols of b rows:")
    #print(cols_order_of_brows)

    if prev == "col":

        for i in range(k):

            index_row_out = order_rows[i] #False index... delete rows in order of increasing values (max:k)
            row_out_real_index = non_del_rows[index_row_out] #Row x = [axI, ... axm... axk]
            b_row_x = values_rows[index_row_out] #axm
            new_del_rows0 = np.append(del_rows, row_out_real_index)
            row_out = first_matrix[index_row_out]
            m = values[i][0][1]



            # print("i = ", i)
            #print("index_row_out",index_row_out)
            #print("Follower's value out", value_out)
            #esto saco
            #print("Del_rows ", del_rows)
            #print("Selected rows ", non_del_rows)
            #print("Row out ", row_out, row_out_real_index)
            #print("new_del_rows0 ", new_del_rows0)
            #print("New del rows0", new_del_rows0)
            #print("Rows")

            for old in range(n-k):  #Old = index row in

                new_del_rows = np.delete(new_del_rows0, [old])
                row_in = np.delete(original_matrix[del_rows[old]], del_cols) #Row y = [CyI, ... Cym... Cyk]
                b_row_y = row_in[m]

                #print("Follower's value in", value_in)
                # esto saco
                #print("Row out", index_row_out, "Row in", del_rows[old])
                #print("Row in", row_in)
                #print()
                #print("New del rows", new_del_rows)
                #print("New selected rows", [i for i in range(n) if i not in new_del_rows])
                #print("Follower's row in",original_matrix[del_rows[old]], row_in)
                #print("values [i]][0][1]",values[i][0][1])

                do_munkres = True

                if b_row_y > b_row_x:

                    q_list = [i for i in range(k) if i != m]

                    for q in q_list:
                        #print("q = ",q)
                        p = rows_order_of_bcols[q]
                        row_p = first_matrix[p]
                        #print("p = ", p)

                        b_row_p = values_rows[p]  #apq = brows_p
                        #b_m = values_rows[index_row_out] #bx of rows = axm = bcols_m
                        old_values_b = b_row_x + b_row_p
                        #print("b_q + b_m ", b_q, "+", b_m, "=", old_values_b)

                        b_row_y = row_in[q] #cyq
                        b_row_p_new = row_p[m]  #apm
                        new_values_b = b_row_y + b_row_p_new
                        #print("b_q' and b_m'",b_q_new,"+", b_m_new, "=", new_values_b)

                        if new_values_b <= old_values_b:
                            #print("Truncated neighbor linear!!")
                            avoided_munkres+=1
                            #print("Do munkres = False")
                            do_munkres = False
                            break

                        else:
                            #do_munkres = True
                            #print("Do munkres = True")

                            s_list = [i for i in range(k) if i != m and i != q]

                            for s in s_list:
                                #print("s= ",s)
                                r = rows_order_of_bcols[s]
                                row_r = first_matrix[r]
                                #print("r= ", r)


                                b_row_r = values_rows[r] #ars ERROR


                                old_values_b = b_row_x + b_row_p + b_row_r

                                b_row_p_new = row_p[s]  # aps
                                b_row_r_new = row_r[m]
                                new_values_b = b_row_y + b_row_p_new + b_row_r_new

                                #print("b_x + b_p + b_r ", b_row_x, "+", b_row_p,
                                      #"+", b_row_r,"=", old_values_b)

                                #print("b_y + b_p' + b_r' ", b_row_y, "+", b_row_p_new,
                                      #"+", b_row_r_new, "=", new_values_b)

                                if new_values_b <= old_values_b:
                                    #print("Truncation in a cuadratic upper bound")
                                    #print("Row in ",del_rows[old],"row out",index_row_out )
                                    avoided_munkres += 1
                                    do_munkres = False
                                    #print("Do munkres = False")
                                    break



                        if do_munkres == False:
                            #print("Truncation")
                            #print("Do munkres = False")
                            break

                    if do_munkres:
                        #print("Do munkres = True")
                        matrix=np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                        #print("Doing munkres")
                        solution = munkres_method(matrix.tolist())[0]
                        munkres+=1

                        #print("munkres", munkres)

                        if solution > current_solution:
                            row_neighbors[tuple(new_del_rows)] = solution
                            #print(matrix.tolist())

                else:
                    avoided_munkres += 1
                    #print("Truncated neighbor constant!!")
                    #print("HM is not applied, cyl <= axl")


            if len(row_neighbors)>0:
                #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                prev = "row"
                break



            else:

                index_col_out = order_cols[i]  # False index... delete rows in order of increasing values (max:k)
                col_out_real_index = non_del_cols[index_col_out]
                b_col_x = values_cols[index_col_out] #b_col_x (value_out)
                new_del_cols0 = np.append(del_cols, col_out_real_index)
                col_out = np.transpose(first_matrix)[index_col_out]
                l = values[i][0][0]

                #print("l = ", l)
                #print("New del col0", new_del_cols0)

                for old in range(n - k):
                    new_del_cols = np.delete(new_del_cols0, [old])
                    col_in = np.delete(np.transpose(original_matrix)[del_cols[old]], del_rows)
                    #print("Follower's col in", np.transpose(original_matrix)[del_cols[old]], col_in)
                    #print("values [i][0][1]", values[i][0][0])
                    b_col_y = col_in[l] #(value_in)

                    do_munkres = True

                    if b_col_y > b_col_x:

                        p_list = [i for i in range(k) if i != l]

                        for p in p_list:

                            q = cols_order_of_brows[p]
                            col_q = np.transpose(first_matrix)[q]

                            b_col_q = values_rows[p] #apq (b_p) try = values_cols[q]
                            #b_l = values_cols[index_col_out]  # bx of rows = axm = bm of cols
                            old_values_b = b_col_x + b_col_q
                            #print("b_p + b_m ", b_p, "+", b_l, "=", old_values_b)

                            b_col_y = col_in[p] #(b_p_new)
                            b_col_q_new = col_q[l] #
                            new_values_b = b_col_y + b_col_q_new
                            #print("b_p' and b_m'", b_p_new, "+", b_l_new, "=", new_values_b)

                            if new_values_b <= old_values_b:
                                #print("Truncated neighbor linear!!")
                                avoided_munkres += 1
                                do_munkres = False
                                break

                            else:
                                r_list = [i for i in range(k) if i != l and i != p]

                                for r in r_list:
                                    s = cols_order_of_brows[r]
                                    col_s = np.transpose(first_matrix)[s]
                                    b_col_s = values_cols[s]

                                    old_values_b = b_col_x + b_col_q + b_col_s

                                    b_col_q_new = col_q[r]
                                    b_col_s_new = col_s[l]
                                    new_values_b = b_col_y + b_col_q_new + b_col_s_new

                                    if new_values_b <= old_values_b:
                                        avoided_munkres += 1
                                        do_munkres = False
                                        break


                            if do_munkres == False:
                                break

                        if do_munkres:
                            matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                            #print("Doing munkres")
                            solution = munkres_method(matrix.tolist())[0]
                            munkres += 1

                            # print("munkres", munkres)


                            if solution > current_solution:
                                col_neighbors[tuple(new_del_cols)] = solution
                                #print("Improvement")
                                #print(matrix.tolist())
                    else:
                        avoided_munkres+=1
                        #print("Truncated neighbor constant!!")
                        #print("HM is not applied, cyl <= axl")

                if len(col_neighbors)>0:
                    #print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                    prev = "col"
                    break

            if len(col_neighbors) > 0:
                break

    else:
        #print("Cols 1st")
        for i in range(k):
            index_col_out = order_cols[i]  # False index... delete rows in order of increasing values (max:k)
            col_out_real_index = non_del_cols[index_col_out]
            b_col_x = values_cols[index_col_out]  # b_col_x (value_out)
            new_del_cols0 = np.append(del_cols, col_out_real_index)
            col_out = np.transpose(first_matrix)[index_col_out]
            l = values[i][0][0]

            # print("l = ", l)
            # print("New del col0", new_del_cols0)

            for old in range(n - k):
                new_del_cols = np.delete(new_del_cols0, [old])
                col_in = np.delete(np.transpose(original_matrix)[del_cols[old]], del_rows)
                # print("Follower's col in", np.transpose(original_matrix)[del_cols[old]], col_in)
                # print("values [i][0][1]", values[i][0][0])
                b_col_y = col_in[l]  # (value_in)

                do_munkres = True

                if b_col_y > b_col_x:

                    p_list = [i for i in range(k) if i != l]

                    for p in p_list:

                        q = cols_order_of_brows[p]
                        col_q = np.transpose(first_matrix)[q]

                        b_col_q = values_rows[p]  # apq (b_p) try = values_cols[q]
                        # b_l = values_cols[index_col_out]  # bx of rows = axm = bm of cols
                        old_values_b = b_col_x + b_col_q
                        # print("b_p + b_m ", b_p, "+", b_l, "=", old_values_b)

                        b_col_y = col_in[p]  # (b_p_new)
                        b_col_q_new = col_q[l]  #
                        new_values_b = b_col_y + b_col_q_new
                        # print("b_p' and b_m'", b_p_new, "+", b_l_new, "=", new_values_b)

                        if new_values_b <= old_values_b:
                            # print("Truncated neighbor linear!!")
                            avoided_munkres += 1
                            do_munkres = False
                            break

                        else:
                            r_list = [i for i in range(k) if i != l and i != p]

                            for r in r_list:
                                s = cols_order_of_brows[r]
                                col_s = np.transpose(first_matrix)[s]
                                b_col_s = values_cols[s]

                                old_values_b = b_col_x + b_col_q + b_col_s

                                b_col_q_new = col_q[r]
                                b_col_s_new = col_s[l]
                                new_values_b = b_col_y + b_col_q_new + b_col_s_new

                                if new_values_b <= old_values_b:
                                    avoided_munkres += 1
                                    do_munkres = False
                                    break

                        if do_munkres == False:
                            break

                    if do_munkres:
                        matrix = np.delete(np.delete(original_matrix, del_rows, 0), new_del_cols, 1)
                        # print("Doing munkres")
                        solution = munkres_method(matrix.tolist())[0]
                        munkres += 1

                        # print("munkres", munkres)

                        if solution > current_solution:
                            col_neighbors[tuple(new_del_cols)] = solution
                            # print("Improvement")
                            # print(matrix.tolist())
                else:
                    avoided_munkres += 1
                    # print("Truncated neighbor constant!!")
                    # print("HM is not applied, cyl <= axl")

            if len(col_neighbors) > 0:
                # print("Improvement found deleting col", new_del_cols[len(new_del_cols)-1])
                prev = "col"
                break



            else:
                # print("i = ", i)
                index_row_out = order_rows[i]  # False index... delete rows in order of increasing values (max:k)
                # print("index_row_out",index_row_out)
                row_out_real_index = non_del_rows[index_row_out]  # Row x = [axI, ... axm... axk]

                b_row_x = values_rows[index_row_out]  # axm

                new_del_rows0 = np.append(del_rows, row_out_real_index)

                row_out = first_matrix[index_row_out]

                # print("Follower's value out", value_out)
                # esto saco
                # print("Del_rows ", del_rows)
                # print("Selected rows ", non_del_rows)
                # print("Row out ", row_out, row_out_real_index)

                # print("new_del_rows0 ", new_del_rows0)

                # print("New del rows0", new_del_rows0)
                # print("Rows")
                #print("m = ", m)

                m = values[i][0][1]


                for old in range(n - k):  # Old = index row in

                    new_del_rows = np.delete(new_del_rows0, [old])
                    row_in = np.delete(original_matrix[del_rows[old]], del_cols)  # Row y = [CyI, ... Cym... Cyk]

                    b_row_y = row_in[m]
                    # print("Follower's value in", value_in)

                    # esto saco
                    # print("Row out", index_row_out, "Row in", del_rows[old])
                    # print("Row in", row_in)
                    # print()
                    # print("New del rows", new_del_rows)
                    # print("New selected rows", [i for i in range(n) if i not in new_del_rows])

                    # print("Follower's row in",original_matrix[del_rows[old]], row_in)
                    # print("values [i]][0][1]",values[i][0][1])
                    do_munkres = True
                    # print("Do munkres = True")
                    if b_row_y > b_row_x:

                        q_list = [i for i in range(k) if i != m]

                        for q in range(k):
                            #print("q = ", q)
                            p = rows_order_of_bcols[q]
                            row_p = first_matrix[p]
                            #print("p = ", p)

                            b_row_p = values_rows[p]  # apq = brows_p
                            # b_m = values_rows[index_row_out] #bx of rows = axm = bcols_m
                            old_values_b = b_row_x + b_row_p
                            # print("b_q + b_m ", b_q, "+", b_m, "=", old_values_b)

                            b_row_y = row_in[q]  # cyq
                            b_row_p_new = row_p[m]  # apm
                            new_values_b = b_row_y + b_row_p_new
                            # print("b_q' and b_m'",b_q_new,"+", b_m_new, "=", new_values_b)

                            if new_values_b <= old_values_b:
                                #print("Truncated neighbor linear!!")
                                avoided_munkres += 1
                                # print("Do munkres = False")
                                do_munkres = False
                                break

                            else:

                                s_list = [i for i in range(k) if i != m and i != q]

                                for s in s_list:
                                    # print("s= ",s)
                                    r = rows_order_of_bcols[s]
                                    row_r = first_matrix[r]
                                    # print("r= ", r)

                                    b_row_r = values_rows[r]  # ars ERROR

                                    old_values_b = b_row_x + b_row_p + b_row_r

                                    b_row_p_new = row_p[s]  # aps
                                    b_row_r_new = row_r[m]
                                    new_values_b = b_row_y + b_row_p_new + b_row_r_new

                                    # print("b_x + b_p + b_r ", b_row_x, "+", b_row_p,
                                    # "+", b_row_r,"=", old_values_b)

                                    # print("b_y + b_p' + b_r' ", b_row_y, "+", b_row_p_new,
                                    # "+", b_row_r_new, "=", new_values_b)

                                    if new_values_b <= old_values_b:
                                        #print("Truncation in a cuadratic upper bound")
                                        # print("Row in ",del_rows[old],"row out",index_row_out )
                                        avoided_munkres += 1
                                        do_munkres = False
                                        # print("Do munkres = False")
                                        break

                            if do_munkres == False:
                                # print("Truncation")
                                # print("Do munkres = False")
                                break

                        if do_munkres:
                            # print("Do munkres = True")
                            matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), del_cols, 1)
                            #print("Doing munkres")
                            solution = munkres_method(matrix.tolist())[0]
                            munkres += 1

                            # print("munkres", munkres)

                            if solution > current_solution:
                                row_neighbors[tuple(new_del_rows)] = solution
                                # print(matrix.tolist())

                    else:
                        avoided_munkres += 1
                        #print("Truncated neighbor constant!!")
                        # print("HM is not applied, cyl <= axl")

                if len(row_neighbors) > 0:
                    # print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                    prev = "row"
                    break

    max_value_row, max_value_col = 0, 0

    if len(row_neighbors)>0:
        max_value_row = max(row_neighbors.values())

    if len(col_neighbors)>0:
        max_value_col = max(col_neighbors.values())


    if max_value_row == 0 and max_value_col == 0:
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0
        #print("Total munkres", munkres)

    elif max_value_row >= max_value_col:
        del_rows = max(row_neighbors, key = row_neighbors.get)
        current_solution = max_value_row
        improvement = 1

    else:
        del_cols = max(col_neighbors, key=col_neighbors.get)
        current_solution = max_value_col
        improvement = 1

    matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)

    #print("Original matrix:\n", original_matrix)
    #print("Deleted rows: ", del_rows)
    #print("Deleted columns: ", del_cols)
    #print("Reduced matrix:\n", matrix)
    #print("Values of the solution for AP: ", values)

    return current_solution, del_rows, del_cols, improvement, prev, munkres, avoided_munkres



def neighborhood_6(current_solution,original_matrix, del_rows, del_cols, prev, munkres, avoided_munkres):
    #print("Neighb 1")
    n = original_matrix.shape[0]
    k = n - len(del_rows)
    good_neighbors = dict()
    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)
    #print("Selected rows: ", non_del_rows)
    #print("Selected cols: ", non_del_cols)
    #print("Deleted rows ", del_rows)
    #print("Number of neigbors to visit: ", 2*(n-k)*k)

    first_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    current_solution, values = munkres_method(first_matrix.tolist())[:]

    order_rows = [i[0][0] for i in values]
    #print("order_rows",order_rows)

    order_cols = [i[0][1] for i in values]
    #print("order_cols", order_cols)
    #del_rows =list(range(n-k))
    #print("Del rows,", del_rows)

    #New part of the code
    #print("List to sort")
    #print(values)

    values_rows_dict = dict()
    values_cols_dict = dict()

    for i in values:
        values_rows_dict[i[0][0]] = i[1]
        values_cols_dict[i[0][1]] = i[1]

    values_rows, values_cols = list(), list()

    for i in range(len(values)):
        values_rows.append(values_rows_dict[i])
        values_cols.append(values_cols_dict[i])

    #print("Final sorted vales rows:")
    #print(values_rows)
    #print("Final sorted vales cols:")
    #print(values_cols)



    for row_out in range(k):

        print("row out ", row_out)
        index_row_out = order_rows[row_out] #False index... delete rows in order of increasing values (max:k)

        row_out = non_del_rows[index_row_out]
        value_out_row = values_rows[index_row_out]

        new_del_rows0 = np.append(del_rows, row_out)
        #print("Follower's value out", value_out)
        #esto saco
        #print("Del_rows ", del_rows)
        #print("Selected rows ", non_del_rows)
        #print("Row out ", row_out)

        #print("new_del_rows0 ", new_del_rows0)


        #print("New del rows0", new_del_rows0)
        #print("Rows")

        for row_in in range(n-k):
            #print("row in ", row_in)
            new_del_rows = np.delete(new_del_rows0, [row_in])
            row_in_array = np.delete(original_matrix[del_rows[row_in]], del_cols)
            value_in_row = row_in_array[values[i][0][1]]
            #print("Follower's value in", value_in)

            # esto saco
            #print("Row in ", del_rows[old])
            #print("New del rows", new_del_rows)
            #print("New selected rows", [i for i in range(n) if i not in new_del_rows])

            for col_out in range(k):
                #print("col out ", col_out)
                index_col_out = order_cols[col_out]  # False index... delete rows in order of increasing values (max:k)

                col_out = non_del_cols[index_col_out]
                value_out_col = values_cols[index_row_out]

                new_del_cols0 = np.append(del_cols, col_out)

                for col_in in range(n-k):

                    new_del_cols = np.delete(new_del_cols0, [col_in])

                    col_in_array = np.delete(np.transpose(original_matrix)[del_cols[col_in]], new_del_rows)
                    value_in_row = row_in_array[values[i][0][1]]

                    matrix = np.delete(np.delete(original_matrix, new_del_rows, 0), new_del_cols, 1)

                    solution = munkres_method(matrix.tolist())[0]
                    munkres += 1
                    #print("munkres", munkres)
                    #print("Solution ", solution)

                    if solution > current_solution:
                        new_del_rows_new_del_cols = new_del_rows + new_del_cols
                        good_neighbors[tuple(new_del_rows_new_del_cols)] = solution
                        del_rows = new_del_rows
                        del_cols = new_del_cols
                        break

                if len(good_neighbors) > 0:
                    #print("Improvement found deleting row", new_del_rows[len(new_del_rows)-1])
                    break

            if len(good_neighbors) > 0:
                break

        if len(good_neighbors) > 0:
            break
            #print("Follower's row in",original_matrix[del_rows[old]], row_in)
            #print("values [i]][0][1]",values[i][0][1])


            #if value_in > value_out:



                #print(matrix.tolist())

            #else:
             #   avoided_munkres += 1
                #print("HM is not applied, cyl <= axl")


    max_value = 0

    if len(good_neighbors)>0:
        max_value = max(good_neighbors.values())
        a = max(good_neighbors, key=good_neighbors.get)
        improvement = 1
        print(a)

    if max_value == 0 :
        #print("Local optimum reached with this neighborhood ", current_solution)
        improvement = 0
        #print("Total munkres", munkres)


    matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)

    #print("Original matrix:\n", original_matrix)
    #print("Deleted rows: ", del_rows)
    #print("Deleted columns: ", del_cols)
    #print("Reduced matrix:\n", matrix)
    #print("Values of the solution for AP: ", values)

    return current_solution, del_rows, del_cols, improvement, prev, munkres, avoided_munkres


def apply_neighb(original_matrix, del_rows, del_cols, start_time, current_solution = 0):
    improvement = 1
    it = -1
    matrix = original_matrix.copy()
    prev = "col"
    munkres = 0
    avoided_munkres = 0
    #current_solution = 0
    while improvement == 1:
        total_time = time.time() - start_time

        if total_time < time_limit:
            new_values = neighborhood_5_final(current_solution, matrix, del_rows, del_cols, prev, munkres, avoided_munkres)
            current_solution, del_rows, del_cols, improvement, prev, munkres, avoided_munkres = new_values[:]
            it+=1
            #if improvement == 1:# and it == 0:
                #print("New value :",current_solution, "It", it + 1)
        else:
            print("Time limit reached")
            break
    print("Local optimum reached:", current_solution, "in", it, "iterations.")
    print("Total munkres:", munkres)
    print("Avoided munkres:", avoided_munkres)

    return current_solution, munkres, del_rows, del_cols, avoided_munkres

def random_swap(original_matrix, del_rows, del_cols, num_swaps_rows, num_swaps_cols, order_rows=False, order_cols=False):


    non_del_rows =np.delete(np.array(list(range(n))), del_rows)
    non_del_cols = np.delete(np.array(list(range(n))), del_cols)

    range_n_minus_k = [i for i in range(len(del_rows))]
    range_k = [i for i in range(len(non_del_rows))]

    #print("Current deleted rows: ", sorted(del_rows))
    #print("Current non deleted rows: ", non_del_rows)

    #print("Current deleted cols: ", sorted(del_cols))
    #print("Current non deleted cols: ", non_del_cols)

    rows_to_add = random.sample(range_n_minus_k, num_swaps_rows)
    cols_to_add = random.sample(range_n_minus_k, num_swaps_cols)

    true_rows_to_add = [del_rows[i] for i in rows_to_add]
    true_cols_to_add = [del_cols[i] for i in cols_to_add]

    if order_rows == False:
        rows_to_del_index = random.sample(range_k, num_swaps_rows)
        cols_to_del_index = random.sample(range_k, num_swaps_cols)

    else:
        rows_to_del_index = [order_rows[i] for i in range(num_swaps_rows)]

        if num_swaps_rows+num_swaps_cols <= len(order_rows):
            cols_to_del_index = [order_cols[i] for i in range(num_swaps_rows, num_swaps_rows+ num_swaps_cols)]
        else:
            cols_to_del_index = [order_cols[i] for i in range(len(order_rows)-num_swaps_cols, len(order_rows))]

    rows_to_del = [non_del_rows[i] for i in rows_to_del_index]
    cols_to_del = [non_del_cols[i] for i in cols_to_del_index]

    #print("Add rows (index, row)", rows_to_add,true_rows_to_add)
    #print("Add cols ", true_cols_to_add)

    #print("Del rows ", rows_to_del_index)
    #print("Del cols ", cols_to_del_index)


    #old_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
    #print("Old matrix\n", old_matrix)

    del_rows = np.append(np.delete(del_rows, rows_to_add), rows_to_del)
    del_cols = np.append(np.delete(del_cols, cols_to_add), cols_to_del)

    #print("New rows to delete: ", sorted(del_rows))
    #print("New cols to delete: ", sorted(del_cols))

    #new_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)

    #print("New matrix after swap\n", new_matrix)

    return del_rows, del_cols

def fixed_variables(del_rows, del_cols, fixed_percent, previous=[], rdm = True, change_opt = False):

    non_del_rows = [i for i in range(n) if i not in del_rows]
    non_del_cols = [i for i in range(n) if i not in del_cols]
    print("deleted rows", del_rows)
    print("deleted cols", del_cols)
    #del_rows = del_rows.tolist()
    #del_cols = del_cols.tolist()
    #division_factor = 1 / fixed_percent



    num_deleted_rows = math.floor(len(del_rows) * fixed_percent)
    num_deleted_cols = num_deleted_rows

    num_fixed_rows = math.floor(len(non_del_rows) * fixed_percent)
    num_fixed_cols = num_fixed_rows

    if not change_opt: #typical fixing only variables that are and are not in the solution

        if rdm: #Here the fixed variables are always random
            rows_to_fix = sorted(random.sample(non_del_rows, num_fixed_rows))
            cols_to_fix = sorted(random.sample(non_del_cols, num_fixed_cols))

            rows_to_delete = sorted(random.sample(del_rows,num_deleted_rows))
            cols_to_delete = sorted(random.sample(del_cols,num_deleted_cols))

        else: #here the fixed are the non fixed before
            prev_rows_to_fix = previous[0]
            prev_cols_to_fix = previous[1]
            prev_rows_to_delete = previous[2]
            prev_cols_to_delete = previous[3]

            rows_to_pick_fix = [i for i in non_del_rows if i not in prev_rows_to_fix]
            cols_to_pick_fix = [i for i in non_del_cols if i not in prev_cols_to_fix]

            if num_fixed_rows >= len(rows_to_pick_fix):
                rest_rows_fix = num_fixed_rows - len(rows_to_pick_fix)
                if rest_rows_fix>0:
                    rdm_rest = random.sample(prev_rows_to_fix, rest_rows_fix)
                    rows_to_fix = sorted(rows_to_pick_fix + rdm_rest)
                else:
                    rows_to_fix = sorted(rows_to_pick_fix)

                #print("Rows to fix, select the ones non fixed previously:", rows_to_fix)

            else:
                rows_to_fix = sorted(random.sample(rows_to_pick_fix, num_fixed_rows))

            if num_fixed_cols >= len(cols_to_pick_fix):
                rest_cols_fix = num_fixed_cols - len(cols_to_pick_fix)
                if rest_cols_fix > 0:
                    rdm_rest = random.sample(prev_cols_to_fix, rest_cols_fix)
                    cols_to_fix = sorted(cols_to_pick_fix + rdm_rest)
                else:
                    cols_to_fix = sorted(cols_to_pick_fix)

                #print("Cols to fix, select the ones non fixed previously:", cols_to_fix)

            else:
                cols_to_fix = sorted(random.sample(cols_to_pick_fix, num_fixed_rows))

            rows_to_pick_delete = [i for i in del_rows if i not in prev_rows_to_delete]
            cols_to_pick_delete = [i for i in del_cols if i not in prev_cols_to_delete]

            if num_deleted_rows >= len(rows_to_pick_delete):
                rest_rows_delete = num_deleted_rows - len(rows_to_pick_delete)
                if rest_rows_delete > 0:
                    rdm_rest = random.sample(prev_rows_to_delete, rest_rows_delete)
                    rows_to_delete = sorted(rows_to_pick_delete + rdm_rest)
                else:
                    rows_to_delete = sorted(rows_to_pick_delete)

                #print("Rows to delete, select the ones non deleted previously:", rows_to_delete)

            else:
                rows_to_delete = sorted(random.sample(rows_to_pick_delete, num_deleted_rows))

            if num_deleted_cols >= len(cols_to_pick_delete):
                rest_cols_delete = num_deleted_cols - len(cols_to_pick_delete)
                if rest_cols_delete > 0:
                    rdm_rest = random.sample(prev_cols_to_delete, rest_cols_delete)
                    cols_to_delete = sorted(cols_to_pick_delete + rdm_rest)
                else:
                    cols_to_delete = sorted(cols_to_pick_delete)

                #print("Cols to delete, select the ones non deleted previously:", cols_to_delete)

            else:
                cols_to_delete = sorted(random.sample(cols_to_pick_delete, num_deleted_cols))

    else: #here it fixes to be in the solution one node that is not in the solution
        deleted_to_fix = 1

        #matrix_row_fix =  np.delete(np.delete(original_matrix, non_del_rows, 0), del_cols, 1)

        #print("Matrix with fixed cols, and deleted rows: ")
        #print(matrix_row_fix)

        #max_mean_rows = [round(np.mean(np.sort(i, axis = 0)[:max(2,math.floor(len(non_del_cols)/2))]),2) for i in matrix_row_fix]
        #print("Max_mean_rows ", max_mean_rows)
        #index_row_to_fix = np.argsort(max_mean_rows)[len(max_mean_rows)-1]
        #rows_to_fix = [del_rows[index_row_to_fix]]

        #print("Row selected to fix ",rows_to_fix)
        rows_to_fix = random.sample(del_rows, deleted_to_fix)



        if num_fixed_rows - deleted_to_fix>0:
            rows_to_fix =sorted(rows_to_fix + random.sample(non_del_rows, num_fixed_rows - deleted_to_fix))

        del_rows_1 = [i for i in del_rows if i not in rows_to_fix]
        rows_to_delete = sorted(random.sample(del_rows_1, num_deleted_rows))

        #Initially the columns are normally selected to fix and delete:
        cols_to_fix = sorted(random.sample(non_del_cols, num_fixed_cols))
        cols_to_delete = sorted(random.sample(del_cols, num_deleted_cols))


    print(num_fixed_rows, "fixed variables o[i]: ",  rows_to_fix)
    print(num_fixed_cols, "fixed variables d[j]: ", cols_to_fix)

    print(num_deleted_rows, "deleted variables o[i]: ",  rows_to_delete)
    print(num_deleted_cols, "deleted variables d[j]: ",  cols_to_delete)


    return rows_to_fix, cols_to_fix, rows_to_delete, cols_to_delete

def create_and_solve_model(n, k, M, costs,greater_0,time_limit,rows_to_fix, cols_to_fix, rows_to_delete, cols_to_delete, output=False):


    print("Solving model with DOCPLEX")
    origins = [i for i in range(n)]
    destins = [i for i in range(n)]
    mdl = Model('BAP')

    mdl.set_time_limit(time_limit)
    #mdl.time_limit = time_limit

    o = mdl.binary_var_dict(origins, name="o")
    d = mdl.binary_var_dict(destins, name="d")

    if greater_0 == True:
        u = mdl.continuous_var_dict(origins, name="u", lb=0)
        v = mdl.continuous_var_dict(destins, name="v", lb=0)

    else:
        u = mdl.continuous_var_dict(origins, name="u", lb=-M)
        v = mdl.continuous_var_dict(destins, name="v", lb=-M)

    mdl.maximize(mdl.sum(u[i] for i in origins) + mdl.sum(v[j] for j in destins))

    for i in origins:
        mdl.add_constraint(u[i] <= o[i] * M)
        #mdl.add_constraint(u[0] == 0)

    for j in destins:
        mdl.add_constraint(v[j] <= d[j] * M)

    for i in origins:
        for j in destins:
            mdl.add_constraint(u[i] + v[j] <= costs[i][j] + M * (2 - o[i] - d[j]))


    for i in rows_to_fix:
        mdl.add_constraint(o[i]==1)

    for j in cols_to_fix:
        mdl.add_constraint(d[j]==1)

    for i in rows_to_delete:
        mdl.add_constraint(o[i]==0)

    for j in cols_to_delete:
        mdl.add_constraint(d[j]==0)

    mdl.add_constraint(mdl.sum(o[i] for i in origins) == k)
    mdl.add_constraint(mdl.sum(d[j] for j in destins) == k)

    print("k = ",k)

    solution = mdl.solve(TimeLimit = time_limit, log_output=output)

    # if output == True:
    # print(mdl.export_to_string())
    if output == True:
        print(solution.display())

    sol_value = round(solution.objective_value)

    print("Solution: ", sol_value)
    sol_status = solution.solve_details.status_code

    if sol_status == 101 or sol_status==1 or sol_status==102:
        print("YES, OPTIMAL FOUND")
        status = 1
    else:
        print("Not confirmed to be optimal")
        status = 0

    time = round(solution.solve_details.time, 2)

    print("time ", time)

    rel_gap = round(solution.solve_details.gap, 4)

    # gap : This property returns the MIP relative gap.
    # from http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.sdetails.html#docplex.mp.sdetails.SolveDetails.mip_relative_gap

    print("Relative gap = ", rel_gap)

    # print("Relative gap = ", rel_gap)
    nb_its = solution.solve_details.nb_iterations

    print("Iterations", nb_its)


    o_values = list(solution.get_value_dict(o,False).keys())
    print("o values")
    print(o_values)
    d_values = list(solution.get_value_dict(d, False).keys())
    print("d values")
    print(d_values)

    return sol_value, time, nb_its, status, rel_gap, o_values, d_values


def results_to_Excel(min_costs, munkres_list, local_search_list,
                     diff_solutions, math_improvement_list,
                     math_results_list, math_counter_list, math_fp_impr_global,
                     selected_oi_dj, avoided_munkres_list):

    min_costs_dict,munkres_dict, local_search_dict, columns = dict(),dict(), dict(),list()
    diff_solutions_dict = dict()
    math_improvement_dict = dict()
    math_results_dict = dict()
    math_counter_dict = dict()
    math_fp_impr_dict = dict()

    selected_oi_dj_dict = dict()

    avoided_munkres_dict = dict()

    inst = list()


    for k_i in ks:
        columns.append("k = " + str(k_i))

    for i in instances:
        inst.append("Inst "+str(i))


    for i in range(len(min_costs)):
        min_costs_dict[columns[i]] = min_costs[i]

    for i in range(len(munkres_list)):
        munkres_dict[columns[i]] = munkres_list[i]
        avoided_munkres_dict[columns[i]] = avoided_munkres_list[i]

    for i in range(len(local_search_list)):
        local_search_dict[columns[i]] = local_search_list[i]
        diff_solutions_dict[columns[i]] = diff_solutions[i]

        if matheuristic:
            math_improvement_dict[columns[i]] = math_improvement_list[i]
            math_results_dict[columns[i]] = math_results_list[i]
            math_counter_dict[columns[i]] = math_counter_list[i]
            math_fp_impr_dict[columns[i]] = math_fp_impr_global[i]

    for i in range(len(selected_oi_dj)):
        selected_oi_dj_dict[inst[i]] = selected_oi_dj[i]

    #print(min_costs_dict)
    df = pd.DataFrame(min_costs_dict, columns = columns)
    df.to_excel (r'C:\Users\franc\OneDrive\Thesis\Results.xlsx', index = False, header = True)

    df1 = pd.DataFrame(munkres_dict, columns=columns)

    df1.to_excel(r'C:\Users\franc\OneDrive\Thesis\Results Munkres.xlsx', index=False, header=True)

    df2 = pd.DataFrame(local_search_dict, columns=columns)

    df2.to_excel(r'C:\Users\franc\OneDrive\Thesis\Local search counts.xlsx', index=False, header=True)

    df3= pd.DataFrame(diff_solutions_dict, columns=columns)

    df3.to_excel(r'C:\Users\franc\OneDrive\Thesis\Number of different results.xlsx', index=False, header=True)

    df4= pd.DataFrame(math_improvement_dict, columns=columns)

    df4.to_excel(r'C:\Users\franc\OneDrive\Thesis\Matheuristic improvements.xlsx', index=False, header=True)

    df5= pd.DataFrame(math_results_dict, columns=columns)

    df5.to_excel(r'C:\Users\franc\OneDrive\Thesis\Matheuristic results.xlsx', index=False, header=True)

    df6= pd.DataFrame(math_counter_dict, columns=columns)

    df6.to_excel(r'C:\Users\franc\OneDrive\Thesis\Matheuristic counter.xlsx', index=False, header=True)

    df7= pd.DataFrame(math_fp_impr_dict, columns=columns)

    df7.to_excel(r'C:\Users\franc\OneDrive\Thesis\Matheuristic fixed % improvements.xlsx', index=False, header=True)

    df8 = pd.DataFrame(selected_oi_dj_dict, columns=inst)

    df8.to_excel(r'C:\Users\franc\OneDrive\Thesis\Selected origins and destinations.xlsx', index=False, header=True)

    df9 = pd.DataFrame(avoided_munkres_dict, columns=columns)

    df9.to_excel(r'C:\Users\franc\OneDrive\Thesis\Avoided munkres.xlsx', index=False, header=True)


#DEFINE SIZE OF THE LEADER AND FOLLOWERS MATRIXES '''

local_search = True
swap = False
fixed_percent_global = 0.70
adaptive_fixed_percent = True
matheuristic = True
greater_0 = True
output = False
n = 10
instances = [1]
ks = [1]
export_results = 0

# ---------------------------------------------------------

if n >=50:
    time_limit = 10
elif n==30:
    time_limit = 5

else: time_limit = 1

if instances == [0]:
    instances = [i + 1 for i in range(10)]

if ks == [0]:
    kmin = int(n/10)
    step = int(n/10)
    kmax = n - step
    if n == 10:
        kmin+=1

    ks = [kmin + step * i for i in range(int((kmax - kmin) / step) + 1)]



print("INFORMATION OF THE CURRENT RUN: ")
if local_search and not swap:
    if matheuristic:
        print("Method applied: LS + Matheuristic")
        print("ui vj >=0: ", greater_0)
        print("Print output of CPLEX: ", output)
        print("Fixed percent is variable: ", adaptive_fixed_percent)

    else:
        print("Method applied: LS")

if swap:
    if matheuristic:
        print("Method applied: ILS + Matheuristic")
        print("ui vj >=0: ", greater_0)
        print("Print output of CPLEX: ", output)
        print("Fixed percent is variable: ", adaptive_fixed_percent)

    else:
        print("Method applied: ILS")


print("n =", n)
print("ks", ks)
print("Instances", instances)
print("Time limit ", time_limit)

print("Export results: ", export_results)
a = 1
if matheuristic:    a = 2
print("Aproximate duration: ", round(time_limit*a*len(ks)*len(instances)/3600,2),"hours")

global_results  = calculate_k_inst(instances, 45, local_search, n, swap, matheuristic)[:]

results,munkres_list, del_rows, del_cols, original_matrix, \
local_search_list, diff_solutions, inst_k_sol__del_rows_del_cols, \
math_improvements_list, math_results_list, math_counter_list,\
    math_fp_impr_global, selected_oi_dj, avoided_munkres_list = global_results[:]


print("Optimal solution variables")
print("inst_k_sol__del_rows_del_cols")
print(inst_k_sol__del_rows_del_cols)

#first delete randomly, then delete the nodes that allowthe lowest values

if export_results ==1:
    results_to_Excel(np.transpose(results), np.transpose(munkres_list), np.transpose(local_search_list),
                     np.transpose(diff_solutions), np.transpose(math_improvements_list), np.transpose(math_results_list),
                     np.transpose(math_counter_list), np.transpose(math_fp_impr_global),
                     selected_oi_dj, np.transpose(avoided_munkres_list))

finish_time_code = round(time.time() - start_time_code, 3)

#print("Original matrix")
#print(original_matrix)

print("All the code took", finish_time_code, "seconds")

'''costs = [[5, 5, 7],
 		   [10, 3, 6],
 		   [4, 4, 5]
 ]
print(optimal_sol(np.array(costs),2))
'''

#print("Optimal matrix:")
#optimal_matrix = np.delete(np.delete(original_matrix, del_rows, 0), del_cols, 1)
#print(optimal_matrix)
#print("Optimal followers selection")
#follower_values = munkres_method(optimal_matrix.tolist())[1]
#for i in follower_values:
 #   print(i)