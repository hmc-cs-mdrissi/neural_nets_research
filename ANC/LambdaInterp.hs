module LambdaInterp where

import ForLambdaCommon
import LambdaParser

import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Function
import Control.Applicative (liftA2)

type Context = Map String Int

--data ProgLambda = UnitLambda | IfL Cmp ProgLambda ProgLambda | ExprL Expr
--                | LetLambda String ProgLambda ProgLambda | LetRecLambda String String ProgLambda ProgLambda 
--                | App App deriving Generic

substExpr :: String -> ProgLambda -> Expr -> Expr
substExpr s l1 (Var str) = if str == s then (Var s) else (Var str)
substExpr s l1 (Const i) = (Const i)
substExpr s l1 (Plus expr1 expr2) = (Plus (substExpr s l1 expr1) (substExpr s l1 expr2))
substExpr s l1 (Minus expr1 expr2) = (Minus (substExpr s l1 expr1) (substExpr s l1 expr2))

substCmp :: String -> ProgLambda -> Cmp -> Cmp
substCmp s l1 (Equal expr1 expr2) = liftA2 (==) (evalExpr store expr1) (evalExpr store expr2)
substCmp s l1 (Le expr1 expr2) = liftA2 (<) (evalExpr store expr1) (evalExpr store expr2)
substCmp s l1 (Ge expr1 expr2) = liftA2 (>) (evalExpr store expr1) (evalExpr store expr2)

subst :: String -> ProgLambda -> ProgLambda -> ProgLambda
subst s l1 UnitLambda = UnitLambda
subst s l1 (IfL c l2 l3) = IfL c (subst s l1 l2) (subst s l1 l3)
subst s l1 (ExprL expr) = 

evalExpr :: Context -> Expr -> Maybe Int
evalExpr store (Var str) = Map.lookup str store
evalExpr store (Const i) = Just i
evalExpr store (Plus expr1 expr2) = liftA2 (+) (evalExpr store expr1) (evalExpr store expr2)
evalExpr store (Minus expr1 expr2) = liftA2 (-) (evalExpr store expr1) (evalExpr store expr2)

evalCmp :: Context -> Cmp -> Maybe Bool
evalCmp store (Equal expr1 expr2) = liftA2 (==) (evalExpr store expr1) (evalExpr store expr2)
evalCmp store (Le expr1 expr2) = liftA2 (<) (evalExpr store expr1) (evalExpr store expr2)
evalCmp store (Ge expr1 expr2) = liftA2 (>) (evalExpr store expr1) (evalExpr store expr2)


evalLambda :: Context -> ProgLambda -> Maybe Int
evalLambda store UnitLambda = Nothing
evalLambda store (IfL c l1 l2) = case (evalCmp store c) of
                                    Just True -> eval store l1
                                    Just False -> eval store l2
                                    _ -> Nothing
evalLambda store (Exprl expr) = evalExpr store expr
evalLambda store (LetLambda str l1 l2) = case (evalLambda store l1) of
                                    Just x -> evalLambda (Map.insert str x store) l2
                                    _ -> Nothing
evalLambda store (LetRecLambda str1 str2 l1 l2) = undefined
