{-# LANGUAGE DeriveGeneric #-}

module ForParser where

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
import ForLambdaCommon

data ProgFor = Assign String Expr | If Cmp ProgFor ProgFor | For String Expr Cmp Expr ProgFor
              | Seq ProgFor ProgFor deriving (Show, Generic)

instance ToJSON ProgFor where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON ProgFor

progForDef :: P.LanguageDef ()
progForDef = emptyDef {P.identStart = letter,
                       P.identLetter = alphaNum,
                       P.opStart = P.opLetter progForDef,
                       P.opLetter = oneOf "+-=;><",
                       P.reservedOpNames = ["=", "-", "==", "+", ">", ";", "<"],
                       P.reservedNames = ["if", "then", "else", "endif", "for", "do", "endfor"]}

lexer :: P.TokenParser ()
lexer = P.makeTokenParser progForDef

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

exprE, exprE_term, varE, constE :: Parser Expr
exprE = exprE_term `chainl1` ((plus *> pure Plus) <|> (minus *> pure Minus))
exprE_term = varE <|> constE
varE = Var <$> identifier
constE = Const <$> P.integer lexer

cmpP :: Parser Cmp
cmpP = try (liftA2 Equal (exprE <* double_equal) exprE) <|>
       try (liftA2 Le (exprE <* le) exprE) <|>
       liftA2 (Ge) (exprE <* ge) exprE

progP, progP_term, ifP, assignP, forP :: Parser ProgFor
progP = progP_term `chainl1` (semicolon *> pure Seq)

ifP = do
        reserved "if"
        cond <- cmpP
        reserved "then"
        if_body <- progP
        reserved "else"
        else_body <- progP
        reserved "endif"
        return $ If cond if_body else_body

assignP = liftA2 Assign (identifier <* equal) exprE

forP = do
        reserved "for"
        var <- identifier
        equal
        e1 <- exprE
        semicolon
        cond <- cmpP
        semicolon
        e2 <- exprE
        reserved "do"
        body <- progP
        reserved "endfor"
        return $ For var e1 cond e2 body

progP_term = forP <|> ifP <|> assignP <|> parens progP

parseProg :: String -> Either ParseError ProgFor
parseProg = parse progP ""
