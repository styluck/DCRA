function [h, dh] = env_l1(x, del, OPTS)

 
z = prox_l1(x, del);
h = sum(sum(abs(z))) + .5*(norm(x - z,'fro')^2)./del;

if nargout > 1
  
    dh = (x - z)./del;

end
