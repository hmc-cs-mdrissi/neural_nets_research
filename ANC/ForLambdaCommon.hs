{-# LANGUAGE DeriveGeneric #-}

module ForLambdaCommon where

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Text.Parsec.Error
import Control.Applicative (liftA2)
import Data.Aeson
import GHC.Generics

data Expr = Var String | Const Int | Plus Expr Expr | Minus Expr Expr deriving Generic
data Cmp = Equal Expr Expr | Le Expr Expr | Ge Expr Expr deriving Generic

is_value :: Expr -> Bool
is_value (Var _) = True
is_value (Const _) = True
is_value _ = False

showExpr :: Int -> Expr -> String
showExpr 0 (Plus x y) = (showExpr 0 x) ++ " + " ++ (showExpr 1 y)
showExpr 0 (Minus x y) = (showExpr 0 x) ++ " - " ++ (showExpr 1 y)
showExpr 0 a = showExpr 1 a

showExpr 1 (Var x) = x
showExpr 1 (Const n) = show n
showExpr 1 a = "(" ++ (showExpr 0 a) ++ ")"

instance Show Expr where
    show = showExpr 0

instance Show Cmp where
    show (Equal x y) = show x ++ " == " ++ show y
    show (Le x y) = show x ++ " < " ++ show y
    show (Ge x y) = show x ++ " > " ++ show y  

instance ToJSON Expr where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Expr

instance ToJSON Cmp where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Cmp

sharedDef :: P.LanguageDef ()
sharedDef = emptyDef {P.identStart = letter,
                      P.identLetter = alphaNum,
                      P.opStart = P.opLetter sharedDef,
                      P.opLetter = oneOf "+-=;><",
                      P.reservedOpNames = ["=", "-", "==", "+", ">", ";", "<"],
                      P.reservedNames = ["if", "then", "else", "endif", "for", "do", "endfor", "let", "letrec",
                                          "in", "unit"]}

lexer :: P.TokenParser ()
lexer = P.makeTokenParser sharedDef

reserved :: String -> Parser ()
reserved = P.reserved lexer

reservedOp :: String -> Parser ()
reservedOp = P.reserved lexer

double_equal :: Parser ()
double_equal = P.reservedOp lexer "=="

equal :: Parser ()
equal = P.reservedOp lexer "="

semicolon :: Parser ()
semicolon = P.reservedOp lexer ";"

plus :: Parser ()
plus = P.reservedOp lexer "+"

minus :: Parser ()
minus = P.reservedOp lexer "-"

le :: Parser ()
le = P.reservedOp lexer "<"

ge :: Parser ()
ge = P.reservedOp lexer ">"

identifier :: Parser String
identifier = P.identifier lexer

parens :: Parser a -> Parser a
parens = P.parens lexer

expr, expr_term, var, constant :: Parser Expr
expr = expr_term `chainl1` ((plus *> pure Plus) <|> (minus *> pure Minus))
expr_term = var <|> constant <|> parens expr
var = Var <$> identifier
constant = Const <$> (fromInteger <$> P.integer lexer)

cmp :: Parser Cmp
cmp = try (liftA2 Equal (expr <* double_equal) expr) <|>
       try (liftA2 Le (expr <* le) expr) <|>
       liftA2 (Ge) (expr <* ge) expr <|> parens cmp
