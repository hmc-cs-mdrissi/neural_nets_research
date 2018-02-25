module LambdaInterp where

import LambdaParser
import Data.Function
import Data.Set (Set)
import qualified Data.Set as Set

free_variables :: LambdaExpressionS -> Set String
free_variables (VariableS x) = Set.singleton x
free_variables (ApplicationS x y) = on Set.union free_variables x y
free_variables (AbstractionS x y) = Set.delete x $ free_variables y

bound_variables :: LambdaExpressionS -> Set String
bound_variables (VariableS x) = Set.empty
bound_variables (ApplicationS x y) = on Set.union bound_variables x y
bound_variables (AbstractionS x y) = Set.insert x $ bound_variables y

fresh_variable :: Set String -> String
fresh_variable bound = fresh_variable_aux bound 1
    where fresh_variable_aux bound n = if (name n) `Set.notMember` bound then name n else fresh_variable_aux bound (n+1)
          name n = "a" ++ show n


evalCBV :: LambdaExpressionS -> Either String LambdaExpressionS
evalCBV (ApplicationS (AbstractionS var expr) val@(AbstractionS _ _)) = evalCBV (substitute expr var val)
evalCBV (ApplicationS v@(AbstractionS _ _) e) = ApplicationS v <$> evalCBV e >>= evalCBV
evalCBV (ApplicationS e1 e2) = ApplicationS <$> evalCBV e1 <*> pure e2 >>= evalCBV
evalCBV v@(AbstractionS _ _) = Right v
evalCBV (VariableS var) = Left $ "Error: Unbound variable " ++ var ++ "."

evalCBV' :: LambdaExpression -> Either String LambdaExpression
evalCBV' expr = convert_to_multiple_arguments <$> (evalCBV . convert_to_one_argument) expr

beta_normalize_one_step :: LambdaExpressionS -> LambdaExpressionS
beta_normalize_one_step x@(VariableS y) = x
beta_normalize_one_step (AbstractionS var expr) = AbstractionS var (beta_normalize_one_step expr)
beta_normalize_one_step (ApplicationS (AbstractionS var expr) val) = on (flip substitute var) beta_normalize_one_step expr val
beta_normalize_one_step (ApplicationS x y) = on ApplicationS beta_normalize_one_step x y


beta_normalize :: LambdaExpressionS -> LambdaExpressionS
beta_normalize x | x == y = x
                 | otherwise = beta_normalize y
                 where y = beta_normalize_one_step x

beta_normalize' :: LambdaExpression -> LambdaExpression
beta_normalize' = lift_to_multiple beta_normalize

eta_normalize_one_step :: LambdaExpressionS -> LambdaExpressionS
eta_normalize_one_step (AbstractionS var (ApplicationS expr (VariableS val))) | var == val && val `Set.notMember` (free_variables expr) = eta_normalize_one_step expr
eta_normalize_one_step (ApplicationS x y) = on (ApplicationS) eta_normalize_one_step x y
eta_normalize_one_step (AbstractionS var expr) = AbstractionS var (eta_normalize_one_step expr)
eta_normalize_one_step x = x

eta_normalize :: LambdaExpressionS -> LambdaExpressionS
eta_normalize x | x == y = x
                | otherwise = eta_normalize y
                where y = eta_normalize_one_step x

eta_normalize' :: LambdaExpression -> LambdaExpression
eta_normalize' = lift_to_multiple eta_normalize

beta_equivalent :: LambdaExpressionS -> LambdaExpressionS -> Bool
beta_equivalent = on (==) beta_normalize

beta_equivalent' :: LambdaExpression -> LambdaExpression -> Bool
beta_equivalent' = on (==) beta_normalize'

alpha_equivalent :: LambdaExpressionS -> LambdaExpressionS -> Bool
alpha_equivalent (VariableS x) (VariableS y) = x == y
alpha_equivalent (AbstractionS x expr) (AbstractionS y expr') = alpha_equivalent (substitute expr x (VariableS new_name)) (substitute expr' y (VariableS new_name))
            where new_name = fresh_variable (on Set.union bound_variables expr expr')
alpha_equivalent (ApplicationS x y) (ApplicationS x' y') = alpha_equivalent x x' && alpha_equivalent y y'
alpha_equivalent _ _ = False

alpha_equivalent' :: LambdaExpression -> LambdaExpression -> Bool
alpha_equivalent' = on alpha_equivalent convert_to_one_argument

alpha_beta_eta_equivalent :: LambdaExpressionS -> LambdaExpressionS -> Bool
alpha_beta_eta_equivalent = on alpha_equivalent (eta_normalize . beta_normalize)

alpha_beta_eta_equivalent' :: LambdaExpression -> LambdaExpression -> Bool
alpha_beta_eta_equivalent' = on alpha_equivalent' (eta_normalize' . beta_normalize')

convert_to_numeral :: LambdaExpressionS -> Either String Int
convert_to_numeral expr = let normal_form = eta_normalize $ beta_normalize expr
                          in if alpha_equivalent normal_form (AbstractionS "x" (VariableS "x")) 
                                then Right 1
                                else startCount normal_form

startCount :: LambdaExpressionS -> Either String Int
startCount (AbstractionS f (AbstractionS val expr)) = continueCount f val expr
startCount x = Left $ "Error: Couldn't extract a number from alleged church numeral."

continueCount :: String -> String -> LambdaExpressionS -> Either String Int
continueCount f val (VariableS x) | x == val = Right 0
                                  | otherwise = Left $ "Error: Couldn't extract a number from alleged church numeral."
continueCount f val (ApplicationS (VariableS func) expr) | f == func = (+1) <$> continueCount f val expr
                                                         | otherwise = Left $ "Error: Couldn't extract a number from alleged church numeral."
continueCount _ _ _ = Left $ "Error: Couldn't extract a number from alleged church numeral."


convert_to_numeral' :: LambdaExpression -> Either String Int
convert_to_numeral' = convert_to_numeral . convert_to_one_argument