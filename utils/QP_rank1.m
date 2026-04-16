function [x] = QP_rank1(b,c,R,gamma)
% min_{x} 0.5*x'*(gamma I+b*b')*x+c'*x, s.t. ||x|| <= R


% min_{x} 0.5*x'*(gamma I+b*b')*x+c'*x, s.t. 0.5 ||x||^2 <= 0.5 R^2
% L(x,theta) = 0.5*x'*(gamma I+b*b')*x+c'*x + theta ( 0.5 ||x||^2 - 0.5 R^2)
% A = gamma I + bb'
% Grad_x = Ax+c + theta x = 0
% (A+theta)x = -c
% x = -inv(A+theta I)c
% max_{theta} 0.5*x'*(A+theta)*x + c'*x - 0.5 theta R^2
% max_{theta} -0.5*x'*c + c'*x - 0.5 theta R^2
% max_{theta} 0.5*x'*c - 0.5 theta R^2
% max_{theta} -0.5*c'*inv(gamma I + bb'+theta I)c - 0.5 theta R^2
% min_{theta} 0.5*c'*inv(b*b'+[gamma+theta]I)c + 0.5 theta R^2

% delta = 0.1;
% b = randn(10,1);
% I  = eye(10);
% inv(delta*I+b*b') - ( 1/delta * I - 1/(delta^2+delta*b'*b) * b*b')

% inv([gamma+theta]I + b*b') = ( 1/(gamma+theta) * I - 1/((gamma+theta)^2+(gamma+theta)*b'*b) * b*b')

% min_{theta} 0.5*c'* [ 1/(gamma+theta) * I - 1/((gamma+theta)^2+(gamma+theta)*b'*b) * b*b' ] c + 0.5 theta R^2

% r = b'*b
% s = c'*c
% t = c'*b

% min_{theta} 0.5*[ c'c*1/(gamma+theta)   -  c'*1/( (gamma+theta)^2+(gamma+theta)*r) * b*b'c ]  + 0.5 theta R^2

% min_{theta} 0.5*[ s/(gamma+theta)   -  tt /( (gamma+theta)^2+(gamma+theta)*r)  ]  + 0.5 theta R^2

% min_{theta} 0.5*s/(gamma+theta)   -  0.5*tt /( (gamma+theta)^2+(gamma+theta)*r)  + 0.5 theta R^2

r = b'*b;
s = c'*c;
t = c'*b;


[theta] = solve1dim(r,s,t,gamma,R);


% ( gamma+theta + r)^2
% dd
% n = length(b);
% x = -inv(eye(n)+b*b'+theta*eye(n))*c;
% [x] = -rank_one_inv(1+theta,b)*c;
delta = gamma+theta;
x = t/(delta^2+delta*r)*b - c/delta;


function [x] = rank_one_inv(delta,b)
% x = inv(delta*I+b*b');
I = eye(length(b));
x = 1/delta * I - 1/(delta^2+delta*b'*b) * b*b'



function [theta] = solve1dim(r,s,t,gamma,R)

fobjHandle = @(theta)  0.5*s/(gamma+theta) + 0.5*theta*R^2 -  0.5*t*t /( (gamma+theta)^2+(gamma+theta)*r);
% fobjHandle = @(theta) 0.5*s   -  0.5*t*t /( (gamma+theta)+r)  + 0.5*theta*(gamma+theta)*R^2;
% theta = fminsearch(fobjHandle,0);

theta = gss(fobjHandle,0,1e100,eps*100,2000,0);


function [a,b] = gss(f,a,b,eps,N,verb)
%
% Performs golden section search on the function f.
% Assumptions: f is continuous on [a,b]; and
% f has only one minimum in [a,b].
% No more than N function evaluations are done.
% When b-a < eps, the iteration stops.
%
% Example: [a,b] = gss('myfun',0,1,0.01,20)
%
c = (-1+sqrt(5))/2;
x1 = c*a + (1-c)*b;
fx1 = feval(f,x1);
x2 = (1-c)*a + c*b;
fx2 = feval(f,x2);
if(verb)
    fprintf('------------------------------------------------------\n');
    fprintf(' x1 x2 f(x1) f(x2) b - a\n');
    fprintf('------------------------------------------------------\n');
    fprintf('%.4e %.4e %.4e %.4e %.4e\n', x1, x2, fx1, fx2, b-a);
end
for i = 1:N-2

    if fx1 < fx2
        b = x2;
        x2 = x1;
        fx2 = fx1;
        x1 = c*a + (1-c)*b;
        fx1 = feval(f,x1);
    else
        a = x1;
        x1 = x2;
        fx1 = fx2;
        x2 = (1-c)*a + c*b;
        fx2 = feval(f,x2);
    end;if(verb)
        fprintf('%.4e %.4e %.4e %.4e %.4e\n', x1, x2, fx1, fx2, b-a);end
    if (abs(b-a) < eps)
        if(verb)
            fprintf('succeeded after %d steps\n', i);
        end
        return;
    end;
end;
if(verb)
    fprintf('failed requirements after %d steps\n', N);
end





