module ForInterp where

import ForLambdaCommon
import ForParser

import Data.Map (Map)
import qualified Data.Map as Map
import Control.Applicative (liftA2)

type Context = Map String Int

evalExpr :: Context -> Expr -> Maybe Int
evalExpr store (Var str) = Map.lookup store str
evalExpr store (Const i) = Just i
evalExpr store (Plus expr1 expr2) = liftA2 (+) (evalExpr store expr1) (evalExpr store expr2)
evalExpr store (Minus expr1 expr2) = liftA2 (-) (evalExpr store expr1) (evalExpr store expr2)

evalCmp :: Context -> Cmp -> Maybe Bool
evalCmp store (Equal expr1 expr2) = liftA2 (==) (evalExpr store expr1) (evalExpr store expr2)
evalCmp store (Le expr1 expr2) = liftA2 (==) (evalExpr store expr1) (evalExpr store expr2)
evalCmp store (Ge expr1 expr2) = liftA2 (==) (evalExpr store expr1) (evalExpr store expr2)


eval :: Context -> ProgFor -> Maybe Context
eval store (Assign str expr) = store.insert str expr
eval store (If c prog1 prog2) = case (evalCmp store c) of
	Just True -> eval store prog1
	Just False -> eval store prog2
	_ -> Nothing