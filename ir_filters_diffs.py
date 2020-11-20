def get_ones(y):
  y = str(int(y))
  y_ones = y[-1]
  return int(y_ones)

def get_tens(y):
  y = str(int(y))
  if len(y) == 1:
    return 0
  elif len(y) == 2:
    return int(y[0])
  else:
    print("Warning: more than two digits")
    return int(y[-2])

def get_p(y, i):
  y = str(int(y))
  if i < len(y):
    return int(y[i])
  else:
    return 0


def get_filters_diffs(custom_name, no_all_pairs, all_source_positions, max_src_len):
  if custom_name == 'custom_adddiv':
    # Filters
    def filt_x1(ti1, ti2, src1, src2):
      return ti1 == 2 and ti2 == 2
    def filt_x2(ti1, ti2, src1, src2):
      return ti1 == 6 and ti2 == 6
    def filt_x3(ti1, ti2, src1, src2):
      return ti1 == 9 and ti2 == 9
    def plus(ti1, ti2, src1, src2):
      token1 = src1[ti1]
      token2 = src2[ti2]
      return token1 == '+' and token2 == '+'
    def plus_left(ti1, ti2, src1, src2):
      both_plus = plus(ti1, ti2, src1, src2)
      return both_plus and src1[2] == src2[2]
    def plus_right(ti1, ti2, src1, src2):
      both_plus = plus(ti1, ti2, src1, src2)
      return both_plus and src1[6] == src2[6]
    def div(ti1, ti2, src1, src2):
      token1 = src1[ti1]
      token2 = src2[ti2]
      return token1 == '/' and token2 == '/'
    def div_den(ti1, ti2, src1, src2):
      both_div = div(ti1, ti2, src1, src2)
      return both_div and src1[9] == src2[9]
    filters = {'x1': filt_x1,
              'x2': filt_x2,
              'x3': filt_x3,
              'plus': plus,
              'plus_left': plus_left,
              'plus_right': plus_right,
              'div': div,
              'div_den': div_den}
    if not no_all_pairs:
      def all_pairs(ti1, ti2, src1, src2):
        return True 
      filters['all_pairs'] = all_pairs

    # Difference functions
    def diff_x1(src1, src2):
      return abs(int(src1[2]) - int(src2[2]))
    def diff_x2(src1, src2):
      return abs(int(src1[6]) - int(src2[6]))
    def diff_x3(src1, src2):
      return abs(int(src1[9]) - int(src2[9]))
    def diff_sums(src1, src2):
      sum1 = int(src1[2]) + int(src1[6])
      sum2 = int(src2[2]) + int(src2[6])
      return abs(sum1 - sum2)
    def diff_y(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      return abs(y1 - y2)
    def diff_y_ones(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      y1_ones = get_ones(y1)
      y2_ones = get_ones(y2)
      return abs(y1_ones - y2_ones)
    def diff_y_tens(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      y1_tens = get_tens(y1)
      y2_tens = get_tens(y2)
      return abs(y1_tens - y2_tens)
    def diff_y_p0(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      y1_p0 = get_p(y1, 0)
      y2_p0 = get_p(y2, 0)
      return abs(y1_p0 - y2_p0)
    def diff_y_p1(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      y1_p1 = get_p(y1, 1)
      y2_p1 = get_p(y2, 1)
      return abs(y1_p1 - y2_p1)

    diff_funcs = {'diff_x1':diff_x1,
                  'diff_x2':diff_x2,
                  'diff_x3':diff_x3,
                  'diff_sums':diff_sums,
                  'diff_y':diff_y,
                  'diff_y_ones':diff_y_ones,
                  'diff_y_tens':diff_y_tens, 
                  'diff_y_p0':diff_y_p0,
                  'diff_y_p1':diff_y_p1}

  elif custom_name == 'custom_muldiv':
    # Filters
    def filt_x1(ti1, ti2, src1, src2):
      return ti1 == 2 and ti2 == 2
    def filt_x2(ti1, ti2, src1, src2):
      return ti1 == 4 and ti2 == 4
    def filt_x3(ti1, ti2, src1, src2):
      return ti1 == 8 and ti2 == 8
    def filt_x4(ti1, ti2, src1, src2):
      return ti1 == 10 and ti2 == 10
    def mul_num(ti1, ti2, src1, src2):
      return ti1 == 3 and ti2 == 3
    def mul_den(ti1, ti2, src1, src2):
      return ti1 == 9 and ti2 == 9
    def div(ti1, ti2, src1, src2):
      return ti1 == 6 and ti2 == 6
    filters = {'x1': filt_x1,
              'x2': filt_x2,
              'x3': filt_x3,
              'x4': filt_x4,
              'mul_num': mul_num,
              'mul_den': mul_den,
              'div': div}
    if not no_all_pairs:
      def all_pairs(ti1, ti2, src1, src2):
        return True 
      filters['all_pairs'] = all_pairs

    # Difference functions
    def diff_x1(src1, src2):
      return abs(int(src1[2]) - int(src2[2]))
    def diff_x2(src1, src2):
      return abs(int(src1[4]) - int(src2[4]))
    def diff_x3(src1, src2):
      return abs(int(src1[8]) - int(src2[8]))
    def diff_x4(src1, src2):
      return abs(int(src1[10]) - int(src2[10]))
    def diff_mul_num(src1, src2):
      mul1 = int(src1[2])*int(src1[4])
      mul2 = int(src2[2])*int(src2[4])
      return abs(mul1 - mul2)
    def diff_mul_den(src1, src2):
      mul1 = int(src1[8])*int(src1[10])
      mul2 = int(src2[8])*int(src2[10])
      return abs(mul1 - mul2)
    def diff_div(src1, src2):
      div1 = (int(src1[2])*int(src1[4]))/(int(src1[8])*int(src1[10]))
      div2 = (int(src2[2])*int(src2[4]))/(int(src2[8])*int(src2[10]))
      return abs(div1 - div2)
    def diff_div_ones(src1, src2):
      y1 = (int(src1[2])*int(src1[4]))/(int(src1[8])*int(src1[10]))
      y2 = (int(src2[2])*int(src2[4]))/(int(src2[8])*int(src2[10]))
      y1_ones = get_ones(y1)
      y2_ones = get_ones(y2)
      return abs(y1_ones - y2_ones)
    def diff_div_tens(src1, src2):
      y1 = (int(src1[2])*int(src1[4]))/(int(src1[8])*int(src1[10]))
      y2 = (int(src2[2])*int(src2[4]))/(int(src2[8])*int(src2[10]))
      y1_tens = get_tens(y1)
      y2_tens = get_tens(y2)
      return abs(y1_tens - y2_tens)
    def diff_y_p0(src1, src2):
      y1 = (int(src1[2])*int(src1[4]))/(int(src1[8])*int(src1[10]))
      y2 = (int(src2[2])*int(src2[4]))/(int(src2[8])*int(src2[10]))
      y1_p0 = get_p(y1, 0)
      y2_p0 = get_p(y2, 0)
      return abs(y1_p0 - y2_p0)
    def diff_y_p1(src1, src2):
      y1 = (int(src1[2])*int(src1[4]))/(int(src1[8])*int(src1[10]))
      y2 = (int(src2[2])*int(src2[4]))/(int(src2[8])*int(src2[10]))
      y1_p1 = get_p(y1, 1)
      y2_p1 = get_p(y2, 1)
      return abs(y1_p1 - y2_p1)
    diff_funcs = {'diff_x1':diff_x1,
                  'diff_x2':diff_x2,
                  'diff_x3':diff_x3,
                  'diff_x4':diff_x4,
                  'diff_mul_num':diff_mul_num,
                  'diff_mul_den':diff_mul_den,
                  'diff_div':diff_div,
                  'diff_y_ones':diff_div_ones,
                  'diff_y_tens':diff_div_tens,
                  'diff_y_p0':diff_y_p0,
                  'diff_y_p1':diff_y_p1}
  elif custom_name == 'custom_addmul':
    # Filters
    def filt_x1(ti1, ti2, src1, src2):
      return ti1 == 1 and ti2 == 1
    def filt_x2(ti1, ti2, src1, src2):
      return ti1 == 5 and ti2 == 5
    def filt_x3(ti1, ti2, src1, src2):
      return ti1 == 7 and ti2 == 7
    def filt_mul(ti1, ti2, src1, src2):
      return ti1 == 6 and ti2 == 6
    def plus(ti1, ti2, src1, src2):
      return ti1 == 3 and ti2 == 3
    def plus_left(ti1, ti2, src1, src2):
      both_plus = plus(ti1, ti2, src1, src2)
      return both_plus and src1[1] == src2[1]
    filters = {'x1': filt_x1,
              'x2': filt_x2,
              'x3': filt_x3,
              'mul': filt_mul,
              'plus': plus,
              'plus_left': plus_left}
    if not no_all_pairs:
      def all_pairs(ti1, ti2, src1, src2):
        return True 
      filters['all_pairs'] = all_pairs

    # Difference functions
    def diff_x1(src1, src2):
      return abs(int(src1[1]) - int(src2[1]))
    def diff_x2(src1, src2):
      return abs(int(src1[5]) - int(src2[5]))
    def diff_x3(src1, src2):
      return abs(int(src1[7]) - int(src2[7]))
    def diff_mul(src1, src2):
      mul1 = int(src1[5])*int(src1[7])
      mul2 = int(src2[5])*int(src2[7])
      return abs(mul1 - mul2)
    def diff_mul2(src1, src2): # bad parse
      mul1 = int(src1[1])*int(src1[5])
      mul2 = int(src2[1])*int(src2[5])
      return abs(mul1 - mul2)
    def diff_mul3(src1, src2): # bad parse
      mul1 = int(src1[1])*int(src1[7])
      mul2 = int(src2[1])*int(src2[7])
      return abs(mul1 - mul2)
    def diff_sum(src1, src2):
      sum1 = int(src1[1]) + int(src1[5])*int(src1[7])
      sum2 = int(src2[1]) + int(src2[5])*int(src2[7])
      return abs(sum1 - sum2)
    def diff_sum_ones(src1, src2):
      y1 = int(src1[1]) + int(src1[5])*int(src1[7])
      y2 = int(src2[1]) + int(src2[5])*int(src2[7])
      y1_ones = get_ones(y1)
      y2_ones = get_ones(y2)
      return abs(y1_ones - y2_ones)
    def diff_sum_tens(src1, src2):
      y1 = int(src1[1]) + int(src1[5])*int(src1[7])
      y2 = int(src2[1]) + int(src2[5])*int(src2[7])
      y1_tens = get_tens(y1)
      y2_tens = get_tens(y2)
      return abs(y1_tens - y2_tens)
    def diff_sum2(src1, src2): # bad parse
      sum1 = int(src1[1]) + int(src1[5])
      sum2 = int(src2[1]) + int(src2[5])
      return abs(sum1 - sum2)
    def diff_sum3(src1, src2): # bad parse
      sum1 = int(src1[1]) + int(src1[7])
      sum2 = int(src2[1]) + int(src2[7])
      return abs(sum1 - sum2)
    def diff_y_p0(src1, src2):
      y1 = int(src1[1]) + int(src1[5])*int(src1[7])
      y2 = int(src2[1]) + int(src2[5])*int(src2[7])
      y1_p0 = get_p(y1, 0)
      y2_p0 = get_p(y2, 0)
      return abs(y1_p0 - y2_p0)
    def diff_y_p1(src1, src2):
      y1 = int(src1[1]) + int(src1[5])*int(src1[7])
      y2 = int(src2[1]) + int(src2[5])*int(src2[7])
      y1_p1 = get_p(y1, 1)
      y2_p1 = get_p(y2, 1)
      return abs(y1_p1 - y2_p1)
    diff_funcs = {'diff_x1':diff_x1,
                  'diff_x2':diff_x2,
                  'diff_x3':diff_x3,
                  'diff_mul':diff_mul,
                  'diff_mul2':diff_mul2,
                  'diff_mul3':diff_mul3,
                  'diff_sum':diff_sum,
                  'diff_sum2':diff_sum2,
                  'diff_sum3':diff_sum3,
                  'diff_y_ones':diff_sum_ones,
                  'diff_y_tens':diff_sum_tens,
                  'diff_y_p0':diff_y_p0,
                  'diff_y_p1':diff_y_p1}
  
  elif custom_name == 'custom_addmuldiv':
    # Filters
    def filt_x1(ti1, ti2, src1, src2):
      return ti1 == 2 and ti2 == 2
    def filt_x2(ti1, ti2, src1, src2):
      return ti1 == 6 and ti2 == 6
    def filt_x3(ti1, ti2, src1, src2):
      return ti1 == 8 and ti2 == 8
    def filt_x4(ti1, ti2, src1, src2):
      return ti1 == 11 and ti2 == 11
    def filt_mul(ti1, ti2, src1, src2):
      return ti1 == 7 and ti2 == 7
    def plus(ti1, ti2, src1, src2):
      return ti1 == 4 and ti2 == 4
    def plus_left(ti1, ti2, src1, src2):
      both_plus = plus(ti1, ti2, src1, src2)
      return both_plus and src1[2] == src2[2]
    def div(ti1, ti2, src1, src2):
      return ti1 == 10 and ti2 == 10
    def div_den(ti1, ti2, src1, src2):
      both_div = div(ti1, ti2, src1, src2)
      return both_div and src1[11] == src2[11]
    filters = {'x1': filt_x1,
               'x2': filt_x2,
               'x3': filt_x3,
               'x4': filt_x4,
               'mul': filt_mul,
               'plus': plus,
               'plus_left': plus_left,
               'div': div,
               'div_den': div_den}
    if not no_all_pairs:
      def all_pairs(ti1, ti2, src1, src2):
        return True 
      filters['all_pairs'] = all_pairs

    # Difference functions
    def diff_x1(src1, src2):
      return abs(int(src1[2]) - int(src2[2]))
    def diff_x2(src1, src2):
      return abs(int(src1[6]) - int(src2[6]))
    def diff_x3(src1, src2):
      return abs(int(src1[8]) - int(src2[8]))
    def diff_x4(src1, src2):
      return abs(int(src1[11]) - int(src2[11]))
    def diff_mul(src1, src2):
      mul1 = int(src1[6])*int(src1[8])
      mul2 = int(src2[6])*int(src2[8])
      return abs(mul1 - mul2)
    def diff_mul2(src1, src2): # bad parse
      mul1 = int(src1[2])*int(src1[6])
      mul2 = int(src2[2])*int(src2[6])
      return abs(mul1 - mul2)
    def diff_mul3(src1, src2): # bad parse
      mul1 = int(src1[2])*int(src1[8])
      mul2 = int(src2[2])*int(src2[8])
      return abs(mul1 - mul2)
    def diff_sum(src1, src2):
      sum1 = int(src1[2]) + int(src1[6])*int(src1[8])
      sum2 = int(src2[2]) + int(src2[6])*int(src2[8])
      return abs(sum1 - sum2)
    def diff_sum2(src1, src2): # bad parse
      sum1 = int(src1[2]) + int(src1[6])
      sum2 = int(src2[2]) + int(src2[6])
      return abs(sum1 - sum2)
    def diff_sum3(src1, src2): # bad parse
      sum1 = int(src1[2]) + int(src1[8])
      sum2 = int(src2[2]) + int(src2[8])
      return abs(sum1 - sum2)
    def diff_div(src1, src2):
      div1 = (int(src1[2]) + int(src1[6])*int(src1[8]))/int(src1[11])
      div2 = (int(src2[2]) + int(src2[6])*int(src2[8]))/int(src2[11])
      return abs(div1 - div2)
    def diff_div_ones(src1, src2):
      y1 = (int(src1[2]) + int(src1[6])*int(src1[8]))/int(src1[11])
      y2 = (int(src2[2]) + int(src2[6])*int(src2[8]))/int(src2[11])
      y1_ones = get_ones(y1)
      y2_ones = get_ones(y2)
      return abs(y1_ones - y2_ones)
    def diff_div_tens(src1, src2):
      y1 = (int(src1[2]) + int(src1[6])*int(src1[8]))/int(src1[11])
      y2 = (int(src2[2]) + int(src2[6])*int(src2[8]))/int(src2[11])
      y1_tens = get_tens(y1)
      y2_tens = get_tens(y2)
      return abs(y1_tens - y2_tens)
    def diff_div2(src1, src2): # different parse (not bad)
      div1 = (int(src1[6])*int(src1[8]))/int(src1[11])
      div2 = (int(src2[6])*int(src2[8]))/int(src2[11])
      return abs(div1 - div2)
    def diff_div3(src1, src2): # different parse (not bad)
      div1 = int(src1[2])/int(src1[11])
      div2 = int(src2[2])/int(src2[11])
      return abs(div1 - div2)
    def diff_y_p0(src1, src2):
      y1 = (int(src1[2]) + int(src1[6])*int(src1[8]))/int(src1[11])
      y2 = (int(src2[2]) + int(src2[6])*int(src2[8]))/int(src2[11])
      y1_p0 = get_p(y1, 0)
      y2_p0 = get_p(y2, 0)
      return abs(y1_p0 - y2_p0)
    def diff_y_p1(src1, src2):
      y1 = (int(src1[2]) + int(src1[6])*int(src1[8]))/int(src1[11])
      y2 = (int(src2[2]) + int(src2[6])*int(src2[8]))/int(src2[11])
      y1_p1 = get_p(y1, 1)
      y2_p1 = get_p(y2, 1)
      return abs(y1_p1 - y2_p1)
    diff_funcs = {'diff_x1':diff_x1,
                  'diff_x2':diff_x2,
                  'diff_x3':diff_x3,
                  'diff_x4':diff_x4,
                  'diff_mul':diff_mul,
                  'diff_mul2':diff_mul2,
                  'diff_mul3':diff_mul3,
                  'diff_sum':diff_sum,
                  'diff_sum2':diff_sum2,
                  'diff_sum3':diff_sum3,
                  'diff_div':diff_div,
                  'diff_div2':diff_div2,
                  'diff_div3':diff_div3,
                  'diff_y_ones':diff_div_ones,
                  'diff_y_tens':diff_div_tens,
                  'diff_y_p0':diff_y_p0,
                  'diff_y_p1':diff_y_p1}
  
  elif custom_name == 'custom_adddivadd':
    # Filters
    def filt_x1(ti1, ti2, src1, src2):
      return ti1 == 2 and ti2 == 2
    def filt_x2(ti1, ti2, src1, src2):
      return ti1 == 6 and ti2 == 6
    def filt_x3(ti1, ti2, src1, src2):
      return ti1 == 9 and ti2 == 9
    def filt_x4(ti1, ti2, src1, src2):
      return ti1 == 13 and ti2 == 13
    def plus1(ti1, ti2, src1, src2):
      return ti1 == 4 and ti2 == 4
    def plus1_left(ti1, ti2, src1, src2):
      both_plus = plus1(ti1, ti2, src1, src2)
      return both_plus and src1[2] == src2[2]
    def plus1_right(ti1, ti2, src1, src2):
      both_plus = plus1(ti1, ti2, src1, src2)
      return both_plus and src1[6] == src2[6]
    def div(ti1, ti2, src1, src2):
      return ti1 == 8 and ti2 == 8
    def div_den(ti1, ti2, src1, src2):
      both_div = div(ti1, ti2, src1, src2)
      return both_div and src1[9] == src2[9]
    def plus2(ti1, ti2, src1, src2):
      return ti1 == 11 and ti2 == 11
    def plus2_right(ti1, ti2, src1, src2):
      both_plus = plus2(ti1, ti2, src1, src2)
      return both_plus and src1[13] == src2[13]
    filters = {'x1': filt_x1,
              'x2': filt_x2,
              'x3': filt_x3,
              'x4': filt_x4,
              'plus1': plus1,
              'plus1_left': plus1_left,
              'plus1_right': plus1_right,
              'div': div,
              'div_den': div_den,
              'plus2': plus2,
              'plus2_right':plus2_right}
    if not no_all_pairs:
      def all_pairs(ti1, ti2, src1, src2):
        return True 
      filters['all_pairs'] = all_pairs

    # Difference functions
    def diff_x1(src1, src2):
      return abs(int(src1[2]) - int(src2[2]))
    def diff_x2(src1, src2):
      return abs(int(src1[6]) - int(src2[6]))
    def diff_x3(src1, src2):
      return abs(int(src1[9]) - int(src2[9]))
    def diff_x4(src1, src2):
      return abs(int(src1[13]) - int(src2[13]))
    def diff_sum1(src1, src2):
      sum1 = int(src1[2]) + int(src1[6])
      sum2 = int(src2[2]) + int(src2[6])
      return abs(sum1 - sum2)
    def diff_div(src1, src2):
      div1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      return abs(div1 - div2)
    def diff_sum2(src1, src2):
      sum1 = (int(src1[2]) + int(src1[6]))/int(src1[9]) + int(src1[13])
      sum2 = (int(src2[2]) + int(src2[6]))/int(src2[9]) + int(src2[13])
      return abs(sum1 - sum2)
    def diff_sum2_ones(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9]) + int(src1[13])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9]) + int(src2[13])
      y1_ones = get_ones(y1)
      y2_ones = get_ones(y2)
      return abs(y1_ones - y2_ones)
    def diff_sum2_tens(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9]) + int(src1[13])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9]) + int(src2[13])
      y1_tens = get_tens(y1)
      y2_tens = get_tens(y2)
      return abs(y1_tens - y2_tens)
    def diff_mul(src1, src2): # different parse (not bad)
      mul1 = int(src1[9])*int(src1[13])
      mul2 = int(src2[9])*int(src2[13])
      return abs(mul1 - mul2)
    def diff_sum3(src1, src2): # different parse (not bad)
      sum1 = int(src1[9])*int(src1[13]) + int(src1[2]) + int(src1[6])
      sum2 = int(src2[9])*int(src2[13]) + int(src2[2]) + int(src2[6])
      return abs(sum1 - sum2)
    def diff_y_p0(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9]) + int(src1[13])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9]) + int(src2[13])
      y1_p0 = get_p(y1, 0)
      y2_p0 = get_p(y2, 0)
      return abs(y1_p0 - y2_p0)
    def diff_y_p1(src1, src2):
      y1 = (int(src1[2]) + int(src1[6]))/int(src1[9]) + int(src1[13])
      y2 = (int(src2[2]) + int(src2[6]))/int(src2[9]) + int(src2[13])
      y1_p1 = get_p(y1, 1)
      y2_p1 = get_p(y2, 1)
      return abs(y1_p1 - y2_p1)
    diff_funcs = {'diff_x1':diff_x1,
                  'diff_x2':diff_x2,
                  'diff_x3':diff_x3,
                  'diff_x4':diff_x4,
                  'diff_sum1':diff_sum1,
                  'diff_div':diff_div,
                  'diff_sum2':diff_sum2,
                  'diff_mul':diff_mul,
                  'diff_sum3':diff_sum3,
                  'diff_y_ones':diff_sum2_ones,
                  'diff_y_tens':diff_sum2_tens,
                  'diff_y_p0':diff_y_p0,
                  'diff_y_p1':diff_y_p1}
  
  elif custom_name == 'custom_adddiv2':
    # Filters
    def filt_x1(ti1, ti2, src1, src2):
      return ti1 == 2 and ti2 == 2
    def filt_x2(ti1, ti2, src1, src2):
      return ti1 == 6 and ti2 == 6
    def filt_x3(ti1, ti2, src1, src2):
      return ti1 == 9 and ti2 == 9
    def filt_x4(ti1, ti2, src1, src2):
      return ti1 == 14 and ti2 == 14
    def filt_x5(ti1, ti2, src1, src2):
      return ti1 == 18 and ti2 == 18
    def filt_x6(ti1, ti2, src1, src2):
      return ti1 == 21 and ti2 == 21
    def term1_plus(ti1, ti2, src1, src2):
      return ti1 == 4 and ti2 == 4
    def term2_plus(ti1, ti2, src1, src2):
      return ti1 == 16 and ti2 == 16
    def term1_div(ti1, ti2, src1, src2):
      return ti1 == 8 and ti2 == 8
    def term2_div(ti1, ti2, src1, src2):
      return ti1 == 20 and ti2 == 20
    def plus(ti1, ti2, src1, src2):
      return ti1 == 11 and ti2 == 11
    filters = {'x1': filt_x1,
               'x2': filt_x2,
               'x3': filt_x3,
               'x4': filt_x4,
               'x5': filt_x5,
               'x6': filt_x6,
               'term1_plus': term1_plus,
               'term2_plus': term2_plus,
               'term1_div': term1_div,
               'term2_div': term2_div,
               'plus': plus}
    if not no_all_pairs:
      def all_pairs(ti1, ti2, src1, src2):
        return True 
      filters['all_pairs'] = all_pairs

    # Difference functions
    def diff_x1(src1, src2):
      return abs(int(src1[2]) - int(src2[2]))
    def diff_x2(src1, src2):
      return abs(int(src1[6]) - int(src2[6]))
    def diff_x3(src1, src2):
      return abs(int(src1[9]) - int(src2[9]))
    def diff_x4(src1, src2):
      return abs(int(src1[14]) - int(src2[14]))
    def diff_x5(src1, src2):
      return abs(int(src1[18]) - int(src2[18]))
    def diff_x6(src1, src2):
      return abs(int(src1[21]) - int(src2[21]))
    def diff_term1_sum(src1, src2):
      sum1 = int(src1[2]) + int(src1[6])
      sum2 = int(src2[2]) + int(src2[6])
      return abs(sum1 - sum2)
    def diff_term2_sum(src1, src2):
      sum1 = int(src1[14]) + int(src1[18])
      sum2 = int(src2[14]) + int(src2[18])
      return abs(sum1 - sum2)
    def diff_term1_div(src1, src2):
      div1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      return abs(div1 - div2)
    def diff_term2_div(src1, src2):
      div1 = (int(src1[14]) + int(src1[18]))/int(src1[21])
      div2 = (int(src2[14]) + int(src2[18]))/int(src2[21])
      return abs(div1 - div2)
    def diff_sum(src1, src2):
      div_left1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div_right1 = (int(src1[14]) + int(src1[18]))/int(src1[21])
      sum1 = div_left1 + div_right1
      div_left2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      div_right2 = (int(src2[14]) + int(src2[18]))/int(src2[21])
      sum2 = div_left2 + div_right2
      return abs(sum1 - sum2)
    def diff_sum_ones(src1, src2):
      div_left1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div_right1 = (int(src1[14]) + int(src1[18]))/int(src1[21])
      y1 = div_left1 + div_right1
      div_left2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      div_right2 = (int(src2[14]) + int(src2[18]))/int(src2[21])
      y2 = div_left2 + div_right2
      y1_ones = get_ones(y1)
      y2_ones = get_ones(y2)
      return abs(y1_ones - y2_ones)
    def diff_sum_tens(src1, src2):
      div_left1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div_right1 = (int(src1[14]) + int(src1[18]))/int(src1[21])
      y1 = div_left1 + div_right1
      div_left2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      div_right2 = (int(src2[14]) + int(src2[18]))/int(src2[21])
      y2 = div_left2 + div_right2
      y1_tens = get_tens(y1)
      y2_tens = get_tens(y2)
      return abs(y1_tens - y2_tens)
    def diff_y_p0(src1, src2):
      div_left1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div_right1 = (int(src1[14]) + int(src1[18]))/int(src1[21])
      y1 = div_left1 + div_right1
      div_left2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      div_right2 = (int(src2[14]) + int(src2[18]))/int(src2[21])
      y2 = div_left2 + div_right2
      y1_p0 = get_p(y1, 0)
      y2_p0 = get_p(y2, 0)
      return abs(y1_p0 - y2_p0)
    def diff_y_p1(src1, src2):
      div_left1 = (int(src1[2]) + int(src1[6]))/int(src1[9])
      div_right1 = (int(src1[14]) + int(src1[18]))/int(src1[21])
      y1 = div_left1 + div_right1
      div_left2 = (int(src2[2]) + int(src2[6]))/int(src2[9])
      div_right2 = (int(src2[14]) + int(src2[18]))/int(src2[21])
      y2 = div_left2 + div_right2
      y1_p1 = get_p(y1, 1)
      y2_p1 = get_p(y2, 1)
      return abs(y1_p1 - y2_p1)
    
    diff_funcs = {'diff_x1':diff_x1,
                  'diff_x2':diff_x2,
                  'diff_x3':diff_x3,
                  'diff_x4':diff_x4,
                  'diff_x5':diff_x5,
                  'diff_x6':diff_x6,
                  'diff_term1_sum':diff_term1_sum,
                  'diff_term2_sum':diff_term2_sum,
                  'diff_term1_div':diff_term1_div,
                  'diff_term2_div':diff_term2_div,
                  'diff_sum':diff_sum,
                  'diff_y_ones':diff_sum_ones,
                  'diff_y_tens':diff_sum_tens,
                  'diff_y_p0':diff_y_p0,
                  'diff_y_p1':diff_y_p1}


  # Additional source filters
  if all_source_positions:
    def filt_position_decorator(p_i):
      def filt(ti1, ti2, src1, src2):
        return ti1 == p_i and ti2 == p_i
      return filt
    for p_i in range(max_src_len):
      filters['p{}'.format(p_i)] = filt_position_decorator(p_i) 

  # Target filters (always the same)
  def trg_filt_same_position(ti1, ti2, trg1, trg2):
    return ti1 == ti2 
  def trg_filt_dig_p0(ti1, ti2, trg1, trg2):
    both_digs = trg1[ti1].isdigit() and trg2[ti2].isdigit()
    return both_digs and ti1 == 0 and ti2 == 0
  def trg_filt_dig_p1(ti1, ti2, trg1, trg2):
    both_digs = trg1[ti1].isdigit() and trg2[ti2].isdigit()
    return both_digs and ti1 == 1 and ti2 == 1
  def trg_filt_eos(ti1, ti2, trg1, trg2):
    return trg1[ti1] == '<eos>' and trg2[ti2] == '<eos>'
  def trg_filt_same_place(ti1, ti2, trg1, trg2):
    neither_eos = ti1 != (len(trg1) - 1) and ti2 != (len(trg2) - 1)
    place1 = len(trg1) - 1 - ti1 
    place2 = len(trg2) - 1 - ti2
    return neither_eos and place1 == place2
  def trg_filt_ones_place(ti1, ti2, trg1, trg2):
    neither_eos = ti1 != (len(trg1) - 1) and ti2 != (len(trg2) - 1)
    place1 = len(trg1) - 1 - ti1 
    place2 = len(trg2) - 1 - ti2
    return neither_eos and place1 == 1 and place2 == 1
  def trg_filt_tens_place(ti1, ti2, trg1, trg2):
    neither_eos = ti1 != (len(trg1) - 1) and ti2 != (len(trg2) - 1)
    place1 = len(trg1) - 1 - ti1 
    place2 = len(trg2) - 1 - ti2
    return neither_eos and place1 == 2 and place2 == 2
  trg_filters = {'trg_filt_position': trg_filt_same_position,
                 'trg_filt_dig_p0': trg_filt_dig_p0,
                 'trg_filt_dig_p1': trg_filt_dig_p1,
                 'trg_filt_eos': trg_filt_eos,
                 'trg_filt_same_place': trg_filt_same_place,
                 'trg_filt_ones_place': trg_filt_ones_place,
                 'trg_filt_tens_place': trg_filt_tens_place}
  if not no_all_pairs:
      def trg_all_pairs(ti1, ti2, src1, src2):
        return True 
      trg_filters['trg_all_pairs'] = trg_all_pairs

  return filters, trg_filters, diff_funcs