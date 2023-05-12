'''
Optimizador L-BFGS adaptado para TensorFlow

Implementacion de PyTorch
https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS
'''

import numpy as np
import tensorflow as tf

from collections import defaultdict


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = np.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = np.abs(d).max()
    g = np.copy(g)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, np.copy(g_new)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if np.abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, np.copy(g_new)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = np.copy(g_new)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if np.abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = np.copy(g_new)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = np.copy(g_new)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def flatten_gradients(grads):

    flat_grads = []
    for g in grads:
        flat_grads.append(np.reshape(g, -1))

    return np.concatenate(flat_grads, axis=0)


class LBFGS():

    '''
    https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
    page 198
    '''

    def __init__(self,
                 params,
                 lr=1.0,
                 max_iter=50,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-10,
                 history_size=200,
                 line_search_fn=None):
        
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        self.defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn)
        
        self._params = params
        self.state = defaultdict(dict)
    
    def _add_grad(self, step_size, update):

        offset = 0
        for p in self._params:
            numel = int(tf.size(p).numpy())
            update_p = np.reshape(update[offset:offset + numel], p.shape) * step_size
            p.assign_add(update_p)
            offset += numel
    
    def _clone_param(self):
        return [p.numpy() for p in self._params]
    
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.assign(pdata)

    def _directional_evaluate(self, closure, args, x, t, d):

        self._add_grad(t, d)
        loss, gradients = closure(*args)
        loss = float(loss.numpy())
        gradients = [tf.convert_to_tensor(g).numpy() for g in gradients]
        flat_grad = flatten_gradients(gradients)
        self._set_param(x)

        return loss, flat_grad

    def step(self, closure, args):

        """Performs a single optimization step.
        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """

        finish = False

        lr = self.defaults['lr']
        max_iter = self.defaults['max_iter']
        max_eval = self.defaults['max_eval']
        tolerance_grad = self.defaults['tolerance_grad']
        tolerance_change = self.defaults['tolerance_change']
        line_search_fn = self.defaults['line_search_fn']
        history_size = self.defaults['history_size']

        state = self.state
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss, gradients = closure(*args)
        orig_loss = orig_loss.numpy()
        gradients = [tf.convert_to_tensor(g).numpy() for g in gradients]

        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = flatten_gradients(gradients)
        opt_cond = np.abs(flat_grad).max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            finish = True
            return orig_loss, finish

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = -1 * flat_grad
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = np.subtract(flat_grad, prev_flat_grad)
                s = np.multiply(d, t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = -1 * flat_grad
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q += (old_dirs[i] * -al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = q.dot(H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r += (old_stps[i] * (al[i] - be_i))

            prev_flat_grad = np.copy(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / np.abs(flat_grad).sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                x_init = self._clone_param()

                def obj_func(x, t, d):
                    return self._directional_evaluate(closure, args, x, t, d)

                loss, flat_grad, t, ls_func_evals = _strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)                
                opt_cond = np.abs(flat_grad).max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    loss, gradients = closure(*args)
                    loss = float(loss.numpy())
                    gradients = [tf.convert_to_tensor(g).numpy() for g in gradients]
                    flat_grad = flatten_gradients(gradients)
                    opt_cond = np.abs(flat_grad).max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################

            # lack of progress
            if np.abs(d.dot(t)).max() <= tolerance_change:
                finish = True
                break


        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss, finish
