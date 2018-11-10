module ForInterp where

import ForLambdaCommon
import ForParser

import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Function
import Control.Applicative (liftA2)

type Context = Map String Int

evalExpr :: Context -> Expr -> Maybe Int
evalExpr store (Var str) = Map.lookup str store
evalExpr store (Const i) = Just i
evalExpr store (Plus expr1 expr2) = liftA2 (+) (evalExpr store expr1) (evalExpr store expr2)
evalExpr store (Minus expr1 expr2) = liftA2 (-) (evalExpr store expr1) (evalExpr store expr2)

evalCmp :: Context -> Cmp -> Maybe Bool
evalCmp store (Equal expr1 expr2) = liftA2 (==) (evalExpr store expr1) (evalExpr store expr2)
evalCmp store (Le expr1 expr2) = liftA2 (<) (evalExpr store expr1) (evalExpr store expr2)
evalCmp store (Ge expr1 expr2) = liftA2 (>) (evalExpr store expr1) (evalExpr store expr2)

evalHelper :: Context -> ProgFor -> Maybe Context
evalHelper store prog@(For var _ cond e2 body) = do cond_value <- evalCmp store cond
                                                    if cond_value
                                                        then do store' <- eval store (Seq body (Assign var e2))
                                                                evalHelper store' prog
                                                        else return store
evalHelper store prog = eval store prog

eval :: Context -> ProgFor -> Maybe Context
eval store (Assign str expr) = flip (Map.insert str) store <$> evalExpr store expr
eval store (If c prog1 prog2) = case (evalCmp store c) of
                                    Just True -> eval store prog1
                                    Just False -> eval store prog2
                                    _ -> Nothing
eval store prog@(For var e1 _ _ _) = do store' <- eval store (Assign var e1)
                                        evalHelper store' prog
eval store (Seq prog1 prog2) = do store' <- eval store prog1
                                  eval store' prog2