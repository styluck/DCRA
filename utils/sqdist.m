function sqd = sqdist(X,Y,w)
% sqd = sqdist(X[,Y,w]) Matrix of squared (weighted) Euclidean distances d(X,Y)

if nargin==1	% Fast version for common case sqdist(X)
  x = sum(X.^2,2); sqd = max(bsxfun(@plus,x,bsxfun(@plus,x',-2*X*X')),0);
  return
end

% ---------- Argument defaults ----------
if ~exist('Y','var') | isempty(Y) Y = X; eqXY = 1; else eqXY=0; end;
% ---------- End of "argument defaults" ----------
  
if exist('w','var') & ~isempty(w)
  h = sqrt(w(:)'); X = bsxfun(@times,X,h);
  if eqXY==1 Y = X; else Y = bsxfun(@times,Y,h); end;
end

% The intervector squared distance is computed as (x-y)� = x�+y�-2xy.
% We ensure that no value is negative (which can happen due to precision loss
% when two vectors are very close).
x = sum(X.^2,2);
if eqXY==1 y = x'; else y = sum(Y.^2,2)'; end;
sqd = max(bsxfun(@plus,x,bsxfun(@plus,y,-2*X*Y')),0);

