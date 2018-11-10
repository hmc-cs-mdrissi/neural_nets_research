{-# LANGUAGE DeriveGeneric #-}

module SimplyTypedLambdaParser where

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Text.Parsec.Expr
import Text.Parsec.Error

import Data.Aeson (ToJSON, FromJSON, toEncoding, genericToEncoding, defaultOptions)
import GHC.Generics (Generic)

import Control.Applicative (some, many, liftA2)
import Data.String
import Data.Function
import Data.Bifunctor
import Data.Functor.Identity

-- TFake is a type for evaluation purposes only when types are not examined.
data Type = TInt | TBool | TIntList | TFun Type Type | TFake deriving (Eq, Generic)

data LambdaExpression = 
    Variable String | -- 1
    Abstraction (String, Type) LambdaExpression | -- 1
    Number Integer | Boolean Bool | Nil | -- Number : 1, Boolean : 11, Nil : 21
    If LambdaExpression LambdaExpression LambdaExpression | -- 11
    Cons LambdaExpression LambdaExpression | -- 21
    Match LambdaExpression LambdaExpression LambdaExpression | -- 21
    UnaryOper UnaryOp LambdaExpression | -- 1
    BinaryOper LambdaExpression BinaryOp LambdaExpression | -- 1 (except no application until 31)
    Let String LambdaExpression LambdaExpression | -- 31
    LetRec (String, Type) LambdaExpression LambdaExpression deriving (Eq, Generic) -- 41

data UnaryOp = Neg | Not deriving (Eq, Generic)
data BinaryOp = Plus | Minus | Times | Divide | And | Or | Equal | Less | Application deriving (Eq, Generic)

instance ToJSON UnaryOp where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON UnaryOp

instance ToJSON BinaryOp where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON BinaryOp

instance ToJSON Type where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Type

instance ToJSON LambdaExpression where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON LambdaExpression

show_variable_list :: [(String, Type)] -> String
show_variable_list [] = ""
show_variable_list [(x,y)] = x ++ " : " ++ show y
show_variable_list ls = unwords $ map (\(x,y) -> "(" ++ x ++ " : " ++ show y ++ ")") ls

check_nat_list :: LambdaExpression -> Bool
check_nat_list Nil = True
check_nat_list (Cons (Number _) l) = check_nat_list l
check_nat_list _ = False

show_nat_list :: LambdaExpression -> String
show_nat_list Nil = "[]"
show_nat_list (Cons (Number n) Nil) = "["  ++ show n ++ "]"
show_nat_list (Cons (Number n) tl) = "[" ++ show n  ++ ";" ++ show_body_nat_list tl ++ "]"
show_nat_list _ = error "Expected nat list."

show_body_nat_list :: LambdaExpression -> String
show_body_nat_list Nil = ""
show_body_nat_list (Cons (Number n) Nil) = show n
show_body_nat_list (Cons (Number n) tl) = show n ++ ";" ++ show_body_nat_list tl
show_body_nat_list _ = error "Expected nat list."

instance Show Type where
    show TInt = "int"
    show TIntList = "list int"
    show TBool = "bool"
    show (TFun x@(TFun _ _) y) = "(" ++ show x ++ ") -> " ++ show y
    show (TFun x y) = show x ++ " -> " ++ show y
    show TFake = "fake_"

instance Show LambdaExpression where
    show = showExpr 0

showExpr :: Int -> LambdaExpression -> String
showExpr 0 x@(Abstraction _ _) = (\(varlist, inner_body) -> "lambda " ++ show_variable_list varlist ++ ". " ++ showExpr 0 inner_body) (variableList x)
    where
        variableList (Abstraction arg expr) = first (arg:) $ variableList expr
        variableList v = ([], v)
showExpr 0 (If x y z) = "if " ++ showExpr 0 x ++ " then " ++ showExpr 0 y ++ " else " ++ showExpr 0 z
showExpr 0 (Let var e e') = "let " ++ var ++ " = " ++ showExpr 0 e ++ " in " ++ showExpr 0 e'
showExpr 0 (LetRec (var, t) e e') = "let rec " ++ var ++ " : " ++ show t ++ " = " ++ showExpr 0 e ++ " in " ++ showExpr 0 e'
showExpr 0 (Match ls e1 (Abstraction (h, _) (Abstraction (t, _) e2))) = "case " ++ showExpr 0 ls ++ " with \ncase [] -> " ++ showExpr 0 e1 ++ "\ncase " ++ h ++ " ; " ++ t ++ " -> " ++ showExpr 0 e2 
showExpr 0 (Match _ _ _) = error "You should not have a match where the second case isn't a double abstraction."
showExpr 0 a = showExpr 1 a

showExpr 1 (BinaryOper x Equal y) = showExpr 2 x ++ " == " ++ showExpr 2 y
showExpr 1 (BinaryOper x Less y) = showExpr 2 x ++ " < " ++ showExpr 2 y
showExpr 1 (BinaryOper (BinaryOper x Less y) Or (BinaryOper a Equal b)) | x == a && y == b = showExpr 2 x ++ " <= " ++ showExpr 2 y
showExpr 1 (UnaryOper Not (BinaryOper x Less y)) = showExpr 2 x ++ " >= " ++ showExpr 2 y
showExpr 1 (UnaryOper Not (BinaryOper (BinaryOper x Less y) Or (BinaryOper a Equal b))) | x == a && y == b = showExpr 2 x ++ " > " ++ showExpr 2 y
showExpr 1 a = showExpr 2 a

showExpr 2 (BinaryOper x Plus y) = showExpr 2 x ++ " + " ++ showExpr 3 y
showExpr 2 (BinaryOper x Minus y) = showExpr 2 x ++ " - " ++ showExpr 3 y
showExpr 2 (BinaryOper (BinaryOper x Less y) Or (BinaryOper a Equal b)) | x == a && y == b = "(" ++ showExpr 2 x ++ " <= " ++ showExpr 2 y ++ ")"
showExpr 2 (BinaryOper x Or y) = showExpr 2 x ++ " or " ++ showExpr 3 y
showExpr 2 a = showExpr 3 a

showExpr 3 (BinaryOper x Times y) = showExpr 3 x ++ " * " ++ showExpr 4 y
showExpr 3 (BinaryOper x Divide y) = showExpr 3 x ++ " / " ++ showExpr 4 y
showExpr 3 (BinaryOper x And y) = showExpr 3 x ++ " and " ++ showExpr 4 y
showExpr 3 a = showExpr 4 a

showExpr 4 (UnaryOper Neg x) = "-" ++ showExpr 5 x
showExpr 4 (UnaryOper Not (BinaryOper x Less y)) = "(" ++ showExpr 2 x ++ " >= " ++ showExpr 2 y ++ ")"
showExpr 4 (UnaryOper Not (BinaryOper (BinaryOper x Less y) Or (BinaryOper a Equal b))) | x == a && y == b = "(" ++ showExpr 2 x ++ " > " ++ showExpr 2 y ++ ")"
showExpr 4 (UnaryOper Not x) = "not " ++ showExpr 5 x
showExpr 4 (BinaryOper x Application y) = showExpr 4 x ++ " " ++ showExpr 5 y
showExpr 4 l@(Cons x y) = if check_nat_list l then show_nat_list l else "cons " ++ showExpr 5 x ++ " " ++ showExpr 5 y 
showExpr 4 a = showExpr 5 a

showExpr 5 (Number n) = show n
showExpr 5 (Boolean True) = "true"
showExpr 5 (Boolean False) = "false"
showExpr 5 (Variable a) = a
showExpr 5 Nil = "[]"
showExpr 5 l@(Cons x y) = if check_nat_list l then show_nat_list l else showExpr 4 l
showExpr _ a = "(" ++ showExpr 0 a ++ ")"

substitute :: LambdaExpression -> String -> LambdaExpression -> LambdaExpression
substitute v@(Variable x) var expr | x == var = expr
                                   | otherwise = v
substitute x@(Abstraction (bound, t) expr1) var expr2 | bound == var =  x
                                                      | otherwise = Abstraction (bound, t) $ substitute expr1 var expr2
substitute x@(Let bound expr1 expr2) var expr3 | bound == var = x
                                               | otherwise = Let bound (substitute expr1 var expr3) (substitute expr2 var expr3)
substitute x@(LetRec (bound,t) expr1 expr2) var expr3 | bound == var = x
                                                      | otherwise = LetRec (bound,t) (substitute expr1 var expr3) (substitute expr2 var expr3)
substitute (If expr1 expr2 expr3) var expr4 = If (substitute expr1 var expr4) (substitute expr2 var expr4) (substitute expr3 var expr4)
substitute (BinaryOper expr1 op expr2) var expr3 = BinaryOper (substitute expr1 var expr3) op (substitute expr2 var expr3)
substitute (UnaryOper op expr1) var expr2 = UnaryOper op $ substitute expr1 var expr2
substitute (Cons expr1 expr2) var expr = Cons (substitute expr1 var expr) (substitute expr2 var expr)
substitute (Match expr1 expr2 expr3) var expr = Match (substitute expr1 var expr) (substitute expr2 var expr) (substitute expr3 var expr)
substitute x _ _ = x

lambdaDef = emptyDef {P.identStart = letter
                     ,P.identLetter = alphaNum
                     ,P.opStart = oneOf "=.;"
                     ,P.opLetter = oneOf "=.;+-/*"
                     ,P.reservedOpNames = ["=", ".", ";", "+", "-", "/", "*", "->", ":", "==", "<", "<=", ">="]
                     ,P.reservedNames = ["lambda", "let", "in", "match","cons", "rec", "int", "bool", "list", "if",
                                         "then", "else", "true", "false", "not", "and", "or", "with", "case"]
                     }

lexer :: P.TokenParser ()
lexer = P.makeTokenParser lambdaDef

reserved :: String -> Parser ()
reserved = P.reserved lexer

reservedOp :: String -> Parser ()
reservedOp = P.reservedOp lexer

identifier :: Parser String
identifier = P.identifier lexer

parens :: Parser a -> Parser a
parens = P.parens lexer

lexeme :: Parser a -> Parser a
lexeme = P.lexeme lexer

whiteSpace :: Parser ()
whiteSpace = P.whiteSpace lexer

listOpen :: Parser ()
listOpen = char '[' *> whiteSpace

listClose :: Parser ()
listClose = char ']' *> whiteSpace

semicolon :: Parser ()
semicolon = reservedOp ";"

basicType :: Parser Type
basicType = (reserved "int" *> pure TInt) <|> (reserved "bool" *> pure TBool)
             <|> (reserved "list" *> reserved "int" *> pure TIntList)

typeP :: Parser Type
typeP = chainr1 (basicType <|> parens typeP) (reservedOp "->" *> pure TFun) <?> "type"

argument :: Parser (String, Type)
argument = liftA2 (,) (identifier <* reservedOp ":") typeP

argumentList :: Parser [(String, Type)]
argumentList = many1 (parens argument) <|>  (pure <$> argument)

variable :: Parser LambdaExpression
variable = Variable <$> identifier

number :: Parser LambdaExpression
number = Number <$> (fromIntegral <$> P.natural lexer)

bool :: Parser LambdaExpression
bool = reserved "true" *> pure (Boolean True) <|> reserved "false" *> pure (Boolean False)

nil :: Parser LambdaExpression
nil = do listOpen
         listClose
         return $ Nil

cons :: Parser LambdaExpression
cons = do reserved "cons"
          n <- lambdaTerm <?> "lambda expression"
          l <- lambdaTerm <?> "lambda expression"
          return $ Cons n l

listHelp :: Parser LambdaExpression
listHelp = lambdaExpression `chainr1` (semicolon *> return Cons)

placeNil :: LambdaExpression -> LambdaExpression
placeNil (Cons x y@(Cons _ _)) = Cons x (placeNil y)
placeNil (Cons x y) = Cons x (Cons y Nil)
placeNil x = Cons x Nil

list :: Parser LambdaExpression
list = try nil <|> placeNil <$> (listOpen *> listHelp <* listClose)

abstraction :: Parser LambdaExpression
abstraction = do reserved "lambda"
                 x <- argumentList
                 reservedOp "."
                 y <- lambdaExpression <?> "lambda expression"
                 return $ createAbstraction x y
    where createAbstraction [] expr = expr
          createAbstraction (arg:rest) expr = Abstraction arg (createAbstraction rest expr)

ifP :: Parser LambdaExpression
ifP = do reserved "if"
         x <- lambdaExpression
         reserved "then"
         y <- lambdaExpression
         reserved "else"
         z <- lambdaExpression
         return $ If x y z

letExpression :: Parser LambdaExpression
letExpression = do reserved "let"
                   var <- identifier
                   reservedOp "="
                   e <- lambdaExpression
                   reserved "in"
                   e' <- lambdaExpression
                   return $ Let var e e'

letRecExpression :: Parser LambdaExpression
letRecExpression = do reserved "let"
                      reserved "rec"
                      arg <- argument
                      reservedOp "="
                      e <- lambdaExpression
                      reserved "in"
                      e' <- lambdaExpression
                      return $ LetRec arg e e'

match :: Parser LambdaExpression
match = do   reserved "match"
             ls <- lambdaExpression <?> "lambda expression"
             reserved "with"
             reserved "case"
             nil
             reservedOp "->"
             e1 <- lambdaExpression
             reserved "case"
             h <- identifier
             reservedOp ";"
             t <- identifier
             reservedOp "->"
             e2 <- lambdaExpression
             return $ Match ls e1 (Abstraction (h, TInt) (Abstraction (t, TIntList) e2))

lambdaTerm :: Parser LambdaExpression
lambdaTerm = number <|> variable <|> bool <|> ifP <|> try letExpression <|> letRecExpression 
                    <|> abstraction <|> list <|> match <|> cons <|> parens lambdaExpression

operatorTable :: OperatorTable String () Identity LambdaExpression
operatorTable = [ [prefixOp "-" (UnaryOper Neg), prefixWord "not" (UnaryOper Not), Infix spaceApp AssocLeft], [binaryOp "*" (flip BinaryOper Times) AssocLeft, 
                   binaryOp "/" (flip BinaryOper Divide) AssocLeft, binaryWord "and" (flip BinaryOper And) AssocLeft], 
                   [binaryOp "+" (flip BinaryOper Plus) AssocLeft, binaryOp "-" (flip BinaryOper Minus) AssocLeft, 
                   binaryWord "or" (flip BinaryOper Or) AssocLeft], [binaryOp "==" (flip BinaryOper Equal) AssocNone,
                   binaryOp "<" (flip BinaryOper Less) AssocNone, binaryOp ">" (\x y -> UnaryOper Not (BinaryOper (BinaryOper x Less y) Or (BinaryOper x Equal y))) AssocNone,
                   binaryOp "<=" (\x y -> BinaryOper (BinaryOper x Less y) Or (BinaryOper x Equal y)) AssocNone,
                   binaryOp ">=" (\x y -> UnaryOper Not (BinaryOper x Less y)) AssocNone]
                   ]

binaryOp, binaryWord :: String -> (LambdaExpression -> LambdaExpression -> LambdaExpression) -> Assoc -> Operator String () Identity LambdaExpression
binaryOp  name fun assoc = Infix (do{ reservedOp name; return fun }) assoc
binaryWord  name fun assoc = Infix (do{ reserved name; return fun }) assoc

prefixOp, postfixOp, prefixWord, postfixWord :: String -> (LambdaExpression -> LambdaExpression) -> Operator String () Identity LambdaExpression
prefixOp  name fun       = Prefix (do{ reservedOp name; return fun })
postfixOp name fun       = Postfix (do{ reservedOp name; return fun })
prefixWord  name fun     = Prefix (do{ reserved name; return fun })
postfixWord name fun     = Postfix (do{ reserved name; return fun })

spaceApp :: Parser (LambdaExpression -> LambdaExpression -> LambdaExpression)
spaceApp = whiteSpace
         *> notFollowedBy (choice . map reservedOp $ P.reservedOpNames lambdaDef)
         *> return (flip BinaryOper Application)

lambdaExpression :: Parser LambdaExpression
lambdaExpression = buildExpressionParser operatorTable lambdaTerm <?> "expression"

parseLambda :: String -> Either ParseError LambdaExpression
parseLambda = parse (whiteSpace *> lambdaExpression <* eof) ""
